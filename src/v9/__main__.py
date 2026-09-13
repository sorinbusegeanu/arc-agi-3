import logging
import warnings

warnings.filterwarnings("ignore")
logging.disable(logging.INFO)

from v9.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
