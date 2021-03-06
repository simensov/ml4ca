from specific.digitwin import DigiTwin
from specific.customEnv import Revolt,RevoltSimple, RevoltLimited, RevoltFinal
from spinup.utils.mpi_tools import proc_id, num_procs
from specific.local_paths import SIM_CONFIG_PATH, SIM_PATH, PYTHON_PORT_INITIAL # this file is not tracked on git due to that it depends on the computer running the code

'''
Examples of my variables from local_paths:
SIM_CONFIG_PATH       = 'C:\\Users\\user\\Documents\\GTK\\configuration'
SIM_PATH              = 'C:\\Users\\user\\Documents\\GTK\\{}\\bin\\revoltsim64.exe'
                        - must be a string like 'C:\\Users\\user\\Documents\\GTK\\{}\\bin\\revoltsim64.exe' so that it can be formatted
PYTHON_PORT_INITIAL   = 25338
USER_DIR              = 'C:\\Users\\user\\GTK\\userdirs\\'
'''

ENVIRONMENTS = {'simple': RevoltSimple, 'limited': RevoltLimited, 'full': Revolt, 'final': RevoltFinal}
class Trainer(object):
    '''
    This class keeps track of all digitwins and its simulators + environments for a training session
    '''
    def __init__(self,
                 n_sims       = 1,
                 start        = True,
                 testing      = False,
                 realtime     = False,
                 simulator_no = 0,
                 lw           = False,
                 env_type     = 'simple',
                 extended_state = False,
                 reset_acts = False,
                 cont_ang = False):

        assert isinstance(n_sims,int) and n_sims > 0, 'Number of simulators must be an integer'
        self._n_sims      = n_sims
        self._digitwins   = []
        self._env_counter = 0
        self._sim_no      = simulator_no # used for running the simulator from different directories
        self._realtime    = realtime
        self._lw          = lw
        self._ext         = extended_state
        self._reset_acts  = reset_acts
        self._cont_ang = cont_ang

        if start:
            self.start_simulators()
            self.set_environments(env_type=env_type,testing = testing)        

    def start_simulators(self,
                         sim_path            = SIM_PATH,
                         python_port_initial = PYTHON_PORT_INITIAL,
                         sim_cfg_path        = SIM_CONFIG_PATH,
                         load_cfg            = False):
        
        ''' Start all simulators '''
        
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

    def set_environments(self,env_type='limited',testing=False):
        n_envs = self._n_sims if num_procs() == 1 else 1 # If multiprocessing, each process only gets one environment
        Env = ENVIRONMENTS[env_type.lower()]
        self._envs = [Env(self._digitwins[i], testing=testing, extended_state=self._ext, reset_acts = self._reset_acts, cont_ang = self._cont_ang) for i in range(n_envs)]

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