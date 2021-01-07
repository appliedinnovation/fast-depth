import torch
import utils

'''

Critical section for each experiment.

An experiment is supposed to test a specific training scenario. The function 
has control over any postprocessing performed on the prediction and target. Each 
function must call the loss function, and then perform any required postprocessing
to prepare the prediction and target for metric evaluation.

The gradients will be backpropagated separately in the training code.
I.e., do not call loss.backward() in the experiment function.

'''


''' 
L1 Loss Experiment

Goal:
    L1 has been shown to be a very stable loss function. It produces adequate results
    and is what was used in the original FastDepth paper. The results are not good 
    enough though, so different experiments with L1 must be tried until it works

Notes:
- Clips to max values before calculating loss

Required JSON params: 
{
    "depth_min" : <min>,
    "depth_max" : <max>,
    "loss" : "silog"
}
'''
class L1LossExperiment():

    def forward(self, prediction, target, criterion, params):

        # Clip prediction to remove zeros and large values
        prediction[prediction < params["depth_min"]] = params["depth_min"]
        prediction[prediction > params["depth_max"]] = params["depth_max"]

        # Calculate L1 loss
        loss = criterion(prediction, target)

        return prediction, target, loss

''' 
Scale-Invariant Log Loss Experiment

Goal:
    Determine if a scale-invariant log loss can produce better training results. 
    This may be possible because with normal L1 loss, distant features are given 
    more importance than nearby features. This is bad because 1) we care more about
    nearby features and 2) distant features are harder to predict accurate scale.
    Ideally, this teaches the model to focus more on nearby features.

Notes:
- Clips to max values before calculating loss

Required JSON params: 
{
    "depth_min" : <min>,
    "depth_max" : <max>,
    "loss" : "silog"
}
'''
class SILogLossExperiment():

    def forward(self, prediction, target, criterion, params):

        # Clip prediction to remove zeros and large values
        prediction[prediction < params["depth_min"]] = params["depth_min"]
        prediction[prediction > params["depth_max"]] = params["depth_max"]

        # Calculate SILog loss
        loss = criterion(prediction, target, interpolate=False)

        return prediction, target, loss
    

# Allows the experiment to be chosen from the parameters file
# Consequently, each experiment must have a unique name
GLOBAL_EXPERIMENT_DICT = {
    "l1_loss_experiment" : L1LossExperiment(),
    "silog_loss_experiment" : SILogLossExperiment()
}

def get_ml_experiment(experiment_str):
    return GLOBAL_EXPERIMENT_DICT[experiment_str]