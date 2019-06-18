import configparser
import sys
import traceback

clf_names = {
            '1': 'MyCanopyKmeans',
            '2': 'StandardCanopyKMeans',
            '3': 'MyKMeans',
            '4': 'SklearnKMeans',
            '5': 'MyKMeans++',
            '6': 'SklearnKMeans++',
            '7': 'KMedoids',
            '8': 'FuzzyCMeans',
        }

def load(config_file, *args):
    """
    Source from my repo: https://github.com/jforjohn/procapi/blob/master/dbod/config.py
    Reads configuration file
    """
    if args:
        args = args[0]
    requiredConfig = {
        'data': ['datadir',
                 'dataset',
        		 'algorithm'],
        'clustering': ['k',
        			   'tol',
        			   'max_rep'],
    }
    try:
        # Load configuration from file
        config = configparser.ConfigParser()
        number_read_files = config.read(config_file)

        # check the config file exist and can be read
        if len(number_read_files) != 1:
            print("Configuration file '{0}' cannot be read or does not exist. Stopping.".format(args.config))
            sys.exit(1)

        # Force load of all required fields to avoid errors in runtime
        for section, options in requiredConfig.items():
            for option in options:
                try:
                    config.get(section, option)
                except configparser.NoSectionError:
                    print("Section '{0}' not present in config file. Stopping.".format(section))
                    sys.exit(2)
                except configparser.NoOptionError:
                    print("Option '{0}' not present in section {1} in config file. Stopping.".format(option, section))
                    sys.exit(2)
        return config

    except IOError as e:
        traceback.print_exc(file=sys.stdout)
        sys.exit(e)
