from Trainer import Trainer
from ReachParameters import parameter_reach_v0, parameter_reach_test
from PushParameters import parameter_push_v0, parameter_push_test
from RiseParameters import parameter_rise_v0, parameter_rise_test
from RiseMultiParameters import parameter_rise_multi_1, parameter_rise_multi_2, parameter_rise_multi_test

TEST = True
GRAPH = False
ENV_TYPES = ["reach", "push", "rise", "rise_multi_worker-stage-1", "rise_multi_worker-stage-2"]
ENV = ENV_TYPES[3]
if __name__ == "__main__":

    # ---------------------------------------- LOAD PARAMETERS ---------------------------------------- #

    if ENV == "reach":
        if not TEST:
            parameters = parameter_reach_v0
        else:
            parameters = parameter_reach_test
    elif ENV == "push":
        if not TEST:
            parameters = parameter_push_v0
        else:
            parameters = parameter_push_test
    elif ENV == "rise":
        if not TEST:
            parameters = parameter_rise_v0
        else:
            parameters = parameter_rise_test
    else:  # RISE MULTI NETWORK (TWO STAGE)
        if not TEST:
            if ENV == "rise_multi_worker-stage-1":
                parameters = parameter_rise_multi_1
            else:
                parameters = parameter_rise_multi_2
        else:
            parameters = parameter_rise_multi_test

    # ---------------------------------------- INITIALIZE TRAINER ---------------------------------------- #

    trainer = Trainer(parameters)
    if parameters["normalize"]:
        trainer.train_normalizer()
    if parameters["load_buffer"]:
        trainer.load_buffer(parameters["buffer_path"])
    if parameters["fill_buffer"]:
        trainer.fill_buffer()
    if GRAPH:
        trainer.graph_training()

    # ---------------------------------------- RUN TRAINER AND SIMULATIONS ---------------------------------------- #

    trainer.run()
