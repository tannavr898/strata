"""One-time Earth Engine authentication for this machine.

WHY this wrapper exists (not just `earthengine authenticate`):
Earth Engine's OAuth performs its code->token exchange through Python's stdlib
`urllib` (OpenSSL), NOT through Strata's `requests` session. On this machine the
local antivirus (Norton) HTTPS-inspects traffic and re-signs certificates with a
private root whose Basic Constraints extension is not marked critical -- which
OpenSSL 3.x strict-rejects, so the token exchange dies with CERTIFICATE_VERIFY_FAILED.
The rest of Strata avoids this by validating against the OS trust store via the
`truststore` package; here we inject truststore GLOBALLY (it patches `ssl`) before
importing `ee`, so Earth Engine's urllib call validates against the Windows store
too. This is the same trust decision the data loaders make -- verification stays ON.

Run once:  .venv/Scripts/python.exe scripts/ee_authenticate.py
A browser opens; approve access. Credentials are saved to your user profile
(~/.config/earthengine/credentials) and reused by every later ee.Initialize().
"""
import truststore

truststore.inject_into_ssl()  # MUST precede `import ee` so its ssl usage is patched

import ee  # noqa: E402

ee.Authenticate()
print("\nEarth Engine authentication complete. You can close the browser tab.")
