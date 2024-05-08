The settings files hold various settings used in the core.py. You can look at that file for how everything is used and all possible settings. Adding a setting can be done in the `core.py` file: 
```python
self.new_setting = self._get('new_setting', 'default_value')
```


The `max_l` setting is used internally for CAMB, it must be larger than lmax by such an amount that the transfer functions fully cover the desired lmax. 


Some settings can be overridden with CLI args, see the `core.py:parse_args` function for more information. 