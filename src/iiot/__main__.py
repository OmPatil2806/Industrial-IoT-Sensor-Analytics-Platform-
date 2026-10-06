"""Allow `python -m iiot ...` as an alternative to the `iiot` command."""

from iiot.cli import main

raise SystemExit(main())
