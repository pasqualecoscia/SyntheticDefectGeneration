from abc import ABC, abstractmethod
import torch

class BaseModel(ABC):
    """Create a model given the options"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda:0" if args.cuda else "cpu")
        
    @abstractmethod
    def create_network():
        """ Create network """
        pass

    @abstractmethod
    def resume_training():
        """ Resume training for each network component if path is provided """
        pass

    @abstractmethod
    def define_losses():
        """ Define loss functions """
        pass

    @abstractmethod
    def set_optimizer():
        """ Set optimizer """
        pass

    @abstractmethod
    def set_decayLR():
        """ Set decay LR"""
        pass

    @abstractmethod
    def prepare_inputs():
        """ Prepare inputs for the model"""
        pass

    @abstractmethod
    def compute_loss_and_update():
        """ Compute model losses and update it"""
        pass

    @abstractmethod
    def set_description():
        """ Define progress bar string output """
        pass

    @abstractmethod
    def update_learning_rates():
        """ Update learning rates """
        pass

    @abstractmethod
    def save_parameters():
        """ Save network parameters """
        pass

    @abstractmethod
    def load_weights():
        """ Load weights for testing """
        pass   

    @abstractmethod
    def set_eval_mode():
        """ Set network in evaluation mode for testing """
        pass     

    @abstractmethod
    def test():
        """ Test model """
        pass   

    @abstractmethod
    def metrics_evaluation():
        """ Evaluate metrics for the model """
        pass   

    @abstractmethod
    def metrics_initialization():
        """ Metrics Initialization """
        pass   

    @abstractmethod
    def print_evaluation_metrics(self):
        """ Print evaluation metrics """
        pass

    @abstractmethod
    def save_training_progress(self):
        """ Save training progress for first batch """
        pass    