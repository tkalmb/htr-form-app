"""appcore -- the thin layer between the Streamlit UI and htrpipe.

Design rule for this package: **no recognition, parsing, preprocessing or
correction logic lives here.** Every such operation is a call into ``htrpipe``
or code taken unchanged from the evaluation implementation that produced
the thesis results (each such block says so in a header comment). The only genuinely new logic is:

* ``inputs.py``       -- turning uploads / PDFs into a directory of page images
* ``plausibility.py`` -- flag-only sanity checks on the extracted values
* ``manifest.py``     -- assembling the run manifest for the app

Everything else is wiring.
"""
