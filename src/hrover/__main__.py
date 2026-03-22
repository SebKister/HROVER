"""Allow running as: python -m hrover [args]

With arguments: runs the CLI (e.g., python -m hrover video.mp4 activity.gpx)
Without arguments: launches the GUI
"""

import sys

if len(sys.argv) > 1:
    from .cli import main
    main()
else:
    from .gui import launch_gui
    launch_gui()
