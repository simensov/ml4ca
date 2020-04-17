from specific.digitwin import DigiTwin
from specific.customEnv import Revolt,RevoltSimple, RevoltLimited
from spinup.utils.mpi_tools import proc_id, num_procs

from specific.local_paths import SIM_CONFIG_PATH, SIM_PATH, PYTHON_PORT_INITIAL
# sim path must be a string like 'C:\\Users\\local\\Documents\\GTK\\{}\\bin\\revoltsim64.exe' so that it can be formatted

ENVIRONMENTS = {'simple': RevoltSimple, 'limited': RevoltLimited, 'full': Revolt}
class Trainer(object):
    '''
    Keeps track of all digitwins and its simulators + environments for a training session
    '''
    def __init__(self,
                 n_sims       = 1,
                 start        = True,
                 testing      = False,
                 realtime     = False,
                 simulator_no = 0,
                 lw           = False,
                 env_type     = 'simple',
                 extended_state = False):

        assert isinstance(n_sims,int) and n_sims > 0, 'Number of simulators must be an integer'
        self._n_sims      = n_sims
        self._digitwins   = []
        self._env_counter = 0
        self._sim_no      = simulator_no # used for running the simulator from different directories
        self._realtime    = realtime
        self._lw          = lw
        self._ext = extended_state

        if start:
            self.start_simulators()
            self.set_environments(env_type=env_type,testing = testing)        

    def start_simulators(self,
                         sim_path            = SIM_PATH,
                         python_port_initial = PYTHON_PORT_INITIAL,
                         sim_cfg_path        = SIM_CONFIG_PATH,
                         load_cfg            = False):
        ''' Start all simulators '''

        # TODO use {}.format() in sim path and rename revoltsim to revoltsim0 for more efficient code
        assert self._sim_no in [0,1,2,3], 'The given sim number is not in the prepared number of simulators'
        
        if self._lw:
            appendix = 'lightweight_revoltsim{}'.format(self._sim_no)
        else:
            appendix = 'revoltsim{}'.format(self._sim_no)

        sim_path = SIM_PATH.format(appendix)
        python_port = PYTHON_PORT_INITIAL + self._sim_no + proc_id()
        print("Open CS sim at: " + str(appendix) + "; Python_port=" + str(python_port))
        self._digitwins.append(None)
        usertag = ('LW{}' if self._lw else '{}').format(self._sim_no)

        self._digitwins[-1] = DigiTwin('Sim'+str(proc_id()), load_cfg, sim_path, sim_cfg_path, python_port, realtime = self._realtime, user = usertag)
        print("Connected to simulators and configuration loaded") if num_procs() == 1 else print('Process {} connected to sim and loaded cfg'.format(proc_id()))

    def get_digitwins(self):
        return self._digitwins

    def set_environments(self,env_type='simple',testing=False):
        n_envs = self._n_sims if num_procs() == 1 else 1 # If multiprocessing, each process only gets one environment
        Env = ENVIRONMENTS[env_type.lower()]
        self._envs = [Env(self._digitwins[i], testing=testing, extended_state=self._ext) for i in range(n_envs)]

    def get_environments(self):
        return self._envs

    def env_fn(self):
        ''' This function is made for returning one environment at a time to the ppo algorithm'''
        if num_procs() > 0:
            # env = self._envs[proc_id()] # TODO old testing from where mpi_fork was called after Trainer()
            env = self._envs[0]
        else:
            env = self._envs[self._env_counter]
            self._env_counter += 1
        return env