import os
import yaml
import logging
import datetime
import random
random.seed(123)
import absl
from absl import app
from absl import flags

from core.utils import Directories, DataLoader
from core.prepare_data import FeatureEngineering
from core.models.ids_models import train as train_ids_models
from core.adv.adv_attacks import craft_ae
from core.adv.plots import run as plot_ae


# from core.adv.plots import Plots


FLAGS = flags.FLAGS
flags.DEFINE_string('root_dir', os.path.normpath(os.getcwd() + os.sep + os.pardir),
                    "root directory for the project")

# --------------------------------- Configuration filenames
flags.DEFINE_string('ip_config', 'ip_config.yaml',
                    "input configuration filename to be used")
flags.DEFINE_string('op_config', 'op_config.yaml',
                    "output configuration filename to be saved")

flags.DEFINE_string('ip_perturb_config', 'ip_perturb_config.yaml',
                    "input perturbation configuration filename to be used")
flags.DEFINE_string('op_perturb_config', 'op_perturb_config.yaml',
                    "output perturbation configuration filename to be saved")

# --------------------------------- Dataset Processing
flags.DEFINE_enum('dataset', 'NSL-KDD', ['NSL-KDD', 'CICDDoS', 'CICIDS'], "name of the dataset?")
flags.DEFINE_boolean('preprocess_data', False, "preprocess the raw dataset?")
flags.DEFINE_boolean('sample_data', False,
                     "create new samples from raw dataset(ONLY in case of CICDDoS)")

# --------------------------------- ML Models
flags.DEFINE_enum('model_config', 'config1', ['config1', 'config2', 'config3', 'config4'],
                  "model configuration to train ML-based IDS models; to be initialized in <ip_config.yaml>")

# --------------------------------- Adversarial Setting

def today() -> str:
    #  used to create log filename based on timestamp
    #  Return DD-MM-YY as a string object
    now = datetime.datetime.now()
    day, month, year = str(now.day), str(now.month), str(now.year)[2:]
    filename = day + '-' + month + '-' + year
    return filename


def set_logger():
    # Override absl logging with base python logging module
    # motivation is to custom create the filename of logs
    filename = today()
    absl.logging.use_python_logging()
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    log_dir = config['dir_config']['logs']
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(filename=os.path.join(log_dir, filename),
                        filemode='w',
                        format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)
    logging.info('=' * 50 + 'Logging Begins' + '=' * 50)
    logging.info('-' * 20 + 'Dataset selected: ' + str(config['dataset']) + '-' * 20)
    print('-' * 20 + 'Dataset selected: ' + str(config['dataset']) + '-' * 20)


def set_config() -> dict:
    '''
    Create a configuration file for different classes and modules of the pipeline.
    config: dict is modified based on *.yaml file in config directory and FLAGS from command-line arguments
    :return: config as dictionary
    '''
    ip_config_filepath = os.path.join(FLAGS.root_dir, 'config', FLAGS.ip_config)
    op_config_filepath = os.path.join(FLAGS.root_dir, 'config', 'tmp', FLAGS.op_config)
    with open(ip_config_filepath, 'r') as stream:
        config = yaml.safe_load(stream)

    config['dataset'] = FLAGS.dataset
    config['sample_data'] = FLAGS.sample_data
    config['preprocess_data'] = FLAGS.preprocess_data
    config['dir_config']['root_dir'] = FLAGS.root_dir
    config['dir_config']['raw_data_dir'] = os.path.join(FLAGS.root_dir, 'data', config['dataset'],
                                                        config['dir_config']['raw_data_dir'], )
    config['dir_config']['processed_data_dir'] = os.path.join(FLAGS.root_dir, 'data', config['dataset'],
                                                              config['dir_config']['processed_data_dir'])
    config['dir_config']['logs'] = os.path.join(FLAGS.root_dir, config['dir_config']['logs'])
    config['dir_config']['results'] = os.path.join(FLAGS.root_dir, config['dir_config']['results'], config['dataset'])
    config['dir_config']['utils'] = os.path.join(FLAGS.root_dir, config['dir_config']['utils'], config['dataset'])
    config['model_config'] = config['model_config'][FLAGS.model_config]

    with open(op_config_filepath, 'w') as stream:
        yaml.dump(config, stream)
    return config


def set_base_directories():
    #  Ensures required base directories are available
    obj = Directories(config)
    obj.make_dirs()  # creates the base directories; if not present already
    obj.make_dirs(os.path.join(obj.results_dir, 'plots'))
    obj.make_dirs(os.path.join(obj.results_dir, 'models'))
    obj.make_dirs(os.path.join(obj.results_dir, 'adv', 'plots'))


def main(argv):

    # FLAGS.dataset = 'NSL-KDD'
    # FLAGS.dataset = 'CICDDoS'
    FLAGS.dataset = 'CICIDS'
    print('Arguments: {}\n{}'.format(argv, '_' * 120))
    global config
    # Order not to be altered
    config = set_config()
    set_logger()  # override absl logging with core python logging module
    set_base_directories()

    if FLAGS.preprocess_data:  # Execution means preprocessing the raw dataset again
        feObj = FeatureEngineering(config)
        feObj.process_raw_data(sample_data_flag=FLAGS.sample_data,
                               save_plots_flag=True)

    # train baseline ml-based ids models
    # train_ids_models(config, save_plots_flag=True)
    n_samples = 2048
    # craft adversarial attacks
    craft_ae(config, n_samples)
    # plots on the adversarial set generated
    # plot_ae(3, 1)
    return 0




if __name__ == '__main__':
    app.run(main)