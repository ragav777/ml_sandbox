import logging
import logging.config

from projects.nfl_ff_pred import nfl_ff_pred as nfl
# from projects.kids_churn import kids_churn as kids
# from projects.telecom_churn import telecom_churn as telecom

##################################################################################################################

# Setup Logging
logging.config.fileConfig('./lib/utils/logging.conf', disable_existing_loggers=False)
log = logging.getLogger('debug')

##################################################################################################################

if __name__ == "__main__":
    # kids.main()
    # telecom.main()
    nfl.main()
