import logging
import numpy as np
import camb

logger = logging.getLogger(__name__)


# Taken from KSW, https://github.com/AdriJD/ksw/blob/master/ksw/cosmo.py, modified some.
# requires code be released at GPL-3.
class Cosmology:
    """
    A Cosmology instance represents a specific cosmology. It is
    used to calculate power spectra and reduced bispectra.

    Taken from KSW, https://github.com/AdriJD/ksw/blob/master/ksw/cosmo.py, with some modifications.

    Parameters
    ----------
    camb_params : camb.model.CAMBparams instance
        CAMB input parameter object.
    verbose : bool, optional
        Report progress.

    Raises
    ------
    ValueError
        If CAMB parameters are invalid.
        If Omega_K parameter is nonzero.

    Attributes
    ----------
    camb_params : camb.model.CAMBparams instance
        Possibly modified copy of input CAMB parameters.
    transfer : dict
        Radiation transfer functions and metadata.
    c_ell : dict
        Angular power spectra and metadata.
    red_bispectra : list
        Collection of ReducedBispectrum instances.
    """

    def __init__(self, core, camb_params, verbose=False):
        self.core = core

        if camb_params.validate() is False:
            raise ValueError("Input CAMB params file invalid.")

        if camb_params.omk != 0:
            raise ValueError("Nonzero Omega_K not supported.")

        self.camb_params = camb_params.copy()

        # Check accuracy and calculation settings, but do not change
        # cosmological or primordial settings.
        self._setattr_camb("WantScalars", True, verbose=verbose)
        self._setattr_camb("WantTensors", False, verbose=verbose)
        self._setattr_camb("WantCls", True, verbose=verbose)
        self._setattr_camb(
            "WantTransfer", False, verbose=verbose
        )  # This is the matter transfer.
        self._setattr_camb("DoLateRadTruncation", False, verbose=verbose)
        self._setattr_camb("DoLensing", True, verbose=verbose)
        self._setattr_camb("AccuracyBoost", 2, subclass="Accuracy", verbose=verbose)
        self._setattr_camb("BessIntBoost", 30, subclass="Accuracy", verbose=verbose)
        self._setattr_camb("KmaxBoost", 3, subclass="Accuracy", verbose=verbose)
        self._setattr_camb("IntTolBoost", 4, subclass="Accuracy", verbose=verbose)
        self._setattr_camb("TimeStepBoost", 4, subclass="Accuracy", verbose=verbose)
        self._setattr_camb(
            "SourcekAccuracyBoost", 5, subclass="Accuracy", verbose=verbose
        )
        self._setattr_camb("BesselBoost", 5, subclass="Accuracy", verbose=verbose)
        self._setattr_camb("IntkAccuracyBoost", 5, subclass="Accuracy", verbose=verbose)
        self._setattr_camb("lSampleBoost", 2, subclass="Accuracy", verbose=verbose)
        self._setattr_camb("lAccuracyBoost", 2, subclass="Accuracy", verbose=verbose)
        self._setattr_camb("AccurateBB", True, subclass="Accuracy", verbose=verbose)
        self._setattr_camb(
            "AccurateReionization", True, subclass="Accuracy", verbose=verbose
        )
        self._setattr_camb(
            "AccuratePolarization", True, subclass="Accuracy", verbose=verbose
        )

        if self.camb_params.validate() is False:
            raise ValueError("Invalid CAMB input")

        self.transfer = {}
        self.c_ell = {}

    def _setattr_camb(self, name, value, subclass=None, verbose=True):
        """
        Set a CAMB parameter and print message if the parameter
        changed its value.

        Parameters
        ----------
        name : str
            Parameter.
        value : obj
            New value.
        subclass : str, optional
            Name of subclass in parameter file.
        verbose : bool, optional
            Print if parameter values change.

        Notes
        -----
        subclass can be: "Accuracy", "NonLinearModel", "DarkEnergy",
            "SourceTerms", "CustomSources", "Reion", "Recomb",
            "InitPower" or "Transfer".

        Raises
        ------
        AttributeError
            If parameter does not exist.
        ValueError
            If new parameter value invalidates parameter object.
        """

        if subclass is not None:
            params = getattr(self.camb_params, subclass)
        else:
            params = self.camb_params

        # CAMB parameter file initiates all its attributes,
        # so we can trust getattr to work here.
        old_value = getattr(params, name)

        if value != old_value:
            setattr(params, name, value)
            if verbose:
                logger.debug(
                    "Updated CAMB param: {} from {} to {}.".format(
                        name, old_value, value
                    )
                )

        if self.camb_params.validate() is False:
            raise ValueError(
                "New value {} for param {} makes params invalid.".format(value, name)
            )

    def compute_transfer(self):
        """
        Call CAMB to calculate radiation transfer functions.

        Parameters
        ----------
        lmax : int
            Maximum multipole.
        verbose : bool, optional
            Print if CAMB parameter values change.

        Raises
        ------
        AttributeError
            If CAMB parameters have not been initialized.
        ValueError
            If lmax is too low (lmax < 300).
        """
        camb_params = self.core.cosmo.camb_params

        k_eta_fac = 2.5  # Default used by CAMB.
        camb_params.set_for_lmax(self.core.max_l, lens_margin=0, k_eta_fac=k_eta_fac)

        # Make CAMB do the actual calculations (slow).
        data = camb.get_transfer_functions(camb_params)
        self._camb_data = data

        # Copy in resulting transfer functions (fast).
        tr = data.get_cmb_transfer_data("scalar")

        # Modify scalar E-mode, see Zaldarriaga 1997 Eqs. 18 and 39.
        # (CAMB applies these factors at a later stage).
        ells = tr.L.astype(int)
        prefactor = np.sqrt((ells + 2) * (ells + 1) * ells * (ells - 1))
        tr.delta_p_l_k[1, ...] *= prefactor[:, np.newaxis]

        # Scale with CMB temperature in uK.
        tr.delta_p_l_k *= camb_params.TCMB * 1e6

        tr_view = tr.delta_p_l_k
        tr_view = np.swapaxes(tr_view, 0, 2)  # (nk, nell, npol).
        tr_view = np.swapaxes(tr_view, 0, 1)  # (nell, nk, npol).

        tr_ell_k = np.ascontiguousarray(tr_view)

        # note transfers are in TT, EE, PHI order
        self.transfer["tr_ell_k"] = tr_ell_k
        self.transfer["k"] = tr.q
        self.transfer["ells"] = ells  # Probably sparse.

    def compute_c_ell(self):
        """
        Calculate angular power spectra (Cls) using precomputed
        transfer functions.

        Notes
        -----
        Spectra are CAMB output and are therefore column-major
        (nell, npol). The pol order is TT, EE, BB, TE. nell can
        be different for lensed and unlensed spectra. The monopole
        and dipole are included. Units are muK^2.
        """
        if not hasattr(self, "_camb_data"):
            logger.warning(
                "Transfer functions not computed before calling computer_c_ell. Computing transfer functions now."
            )
            self.compute_transfer()
        self._camb_data.power_spectra_from_transfer()

        if self.core.lensing:
            c_ell = self._camb_data.get_lensed_scalar_cls(
                CMB_unit="muK",
                raw_cl=True,
            )
        else:
            c_ell = self._camb_data.get_unlensed_scalar_cls(
                CMB_unit="muK",
                raw_cl=True,
            )

        self.c_ell = {}
        self.c_ell["ells"] = np.arange(c_ell.shape[0])
        self.c_ell["c_ell"] = c_ell
