class Config(object):

    # NAS Parameters
    net_layers   = 2
    net_width    = 2
    net_stacking = 3
    net_merging = False
    max_trainable_params = 100000*1

    # RL parameters
    max_num_episodes   = 2000 # i.e. number of epochs for the actor network
    level_episode_step = 10
    reg_param          = 0.001
    reset_state        = False

    # Optmization parameters to train the Network's model
    max_epochs       = 0.05 # it can be < 1 for early stopping
    epochs_increase  = 2
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
    output_dir    = "c:\\tmp\\output\\"+jobID
    checkpointing = 0

