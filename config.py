''''''
class Config(object):

    # NAS Parameters
    net_layers = 3
    net_width  = 3
    net_merging = True

    # RL parameters
    max_num_episodes  = 2000 # i.e. number of epochs
    reg_param         = 0.001

    # Optmization parameters to train the Network's model
    max_epochs       = 1
    steps_per_action = 30000 # i.e. total of sgd steps (or 0 to run all epochs)
    batch_size       = 100
    optimizer        = "rmsprop"

    # Learning rate decay schedule
    learning_rate       = 1e-3
    learning_rate_decay = 0.975
    learning_rate_step  = 500
    learning_rate_min   = 1e-5

    # HCP | Cloud
    jobID = ""

    # Persistence
    output_dir    = "./output/"+jobID+"/"
    checkpointing = 0

