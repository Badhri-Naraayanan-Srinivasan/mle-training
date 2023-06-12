import logging
import logging.config
import os


def configure_logger(
    logger=None, log_file=None, console=False, log_level="INFO"
):
    """Function to setup configurations of logger through function.

    The individual arguments of `log_file`, `console`, `log_level`
    will overwrite the ones in cfg.

    Parameters
    ----------
            logger:
                    Predefined logger object if present. If None a new logger object
                    will be created from root.
            log_file: str
                    Path to the log file for logs to be stored
            console: bool
                    To include a console handler(logs printing in console)
            log_level: str
                    One of `["INFO","DEBUG","WARNING","ERROR","CRITICAL"]`
                    default - `"INFO"`

    Returns
    -------
    logging.Logger
        A custom logger.
    """

    logger = logger or logging.getLogger()

    if log_file or console:
        for hdlr in logger.handlers:
            logger.removeHandler(hdlr)

        if log_file:
            fh = logging.FileHandler(log_file)
            logger.setLevel(getattr(logging, log_level))
            # Create formatters and add it to handlers
            f_format = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%d-%b-%Y %H:%M:%S",
            )
            fh.setFormatter(f_format)
            logger.addHandler(fh)

        if console:
            sh = logging.StreamHandler()
            sh.setLevel(getattr(logging, log_level))
            logger.addHandler(sh)

    return logger


if __name__ == "__main__":
    # configuring and assigning in the logger can be done by the below function
    logger = configure_logger(
        log_file=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "custom_config.log"
        )
    )
    logger.info("Logging Test - Start")
    logger.info("Logging Test - Test 1 Done")
    logger.warning("Watch out!")
