# Loguru issues

### Error:
```
	terminate called after throwing an instance of 'std::runtime_error'
	what(): locale::facet::_S_create_c_locale name not valid
```
### Fix:
#### If first fix doesn't work, go for second fix
* First Fix:
	- Open `~/.profile`
		```
			gedit ~/.profile
		```
	- Add the following line
		```
			export LC_ALL=C; unset LANGUAGE
		```
	- Save and Reboot.
* Second Fix (Requires SUDO):
	- Open `/etc/locale.gen`
		```
			gedit /etc/locale.gen
		```
	- Uncomment the following line:
		```
			en_US.UTF-8 UTF-8
		```
	- Save(sudo) and Reboot.

### If the above methods doesn't fix the error, open a new issue.

# Reference
**[GithHub Issues](https://github.com/potree/PotreeConverter/issues/281)**