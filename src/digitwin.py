# -*- coding: utf-8 -*-
"""
Created on Mon May  7 14:11:34 2018

@author: jonom

Comment Simen: This can be run on both ubuntu and window, but from ubuntu would probably best due to terminal. Just use IP or UDP.
"""

from py4j.java_gateway import JavaGateway, Py4JNetworkError
from py4j.java_gateway import GatewayParameters
import subprocess
import os
import time

from log import log, forcelog

class DigiTwin:
    SCAL_INP = 'scalar_input'
    SCAL_OUT ='scalar_output'
    SCAL_PARA = 'scalar_para'
    VEC_INP = 'vector_input'
    VEC_OUT = 'vector_output'
    VEC_PARA = 'vector_para'
    MODULE_FEAT_TYPES = [SCAL_INP, SCAL_OUT, \
                         VEC_INP, VEC_OUT, \
                         SCAL_PARA, VEC_PARA]
    

    
    
    def __init__(self, name, load_cfg, sim_path, cfg_path, user_dir, web_server_port, python_port):
        self.name = name

        #Open simulator
        self.load_cfg = load_cfg
        self.connectToJVM(sim_path, user_dir, web_server_port, python_port)        

        #setup the function pointers to the python/java interface
        self.mod_feat_func = self.get_mod_feat_funcs()

        #Loading and reading simulator config   
        self.cfg_path = cfg_path
        self.config = self.load_config()

        #step sim as fast as possible
        self.setRealTimeMode(False)
    
####
#method to read or write values to module feature (feat) parameters in the digital twin (simulator)
#no val indicates a read request
#report flag indicates if written val should be printed to std out.
#some features are vectors.  if val = list, len(val) indices of this vector will be written
# if val = number, all indices of this vecor will be written the same value.
#'report' prints out the read/written value if True
#'override' = True use override flag and force value on input values (default for output values)
#'release' = True to release the override.  release cancel an override any time
    def val(self, module, feat, val=None, **args):
        report = args.get('report', False)
        override = args.get('override', False)
        release = args.get('release', False)

        if  'StateResetSingleTargetShip' in feat:
            forcelog(feat+ ' ' + str(val))
            
        single_vec_ix = False
        if "[" in feat:
            content = feat.split("[")
            feat = content[0]
            content = content[1].split("]")
            vec_ix = int(content[0])
            single_vec_ix = True
        ftype = self.get_mod_feat_type(module, feat)
        if ftype is None:
            #if feat != 'StateResetOn':
            forcelog('unknown module or feature !! ' + module + '.' + feat)
            if val is None:
                return None
            return
        if val is None: #a read operation
            val = self.mod_feat_func[ftype][0](module, feat)
            if report:
                forcelog(str(self.name)+' read '+str(val)+' from '+str(module)+'.'+str(feat))
            return val
        else:  #a write operation
            if 'output' in ftype:
                override = True
                
            #is val a vector?
            try:
                valisvec = False
                len(val)
                valisvec = True
            except:
                if type(val) != float:
                    val = float(val)
            if not single_vec_ix: #scalar or full vector
                if 'vector' in ftype:
                    maxveclen = len(self.val(module, feat))
                    if not valisvec:
                        val = [val]*maxveclen
                    else:
                        if len(val) > maxveclen:
                            forcelog('Val vector len ('+str(len(val))+'>'+str(maxveclen)+\
                                                      ') too long for '+module+'.'+feat)
                            val = val[:maxveclen]
                    vec_ix = 0
                    for v in val:
                        if not override:
                            if release:
                                forcelog('(1) release of override on single input/output vector index not implemented!!!')
                            self.mod_feat_func[ftype][2](module, feat, vec_ix, v)
                        else:  #override
                            if release:
                                forcelog('(2) release of override on single input/output vector index not implemented!!!')
                            else:
                                forcelog('(3) override single input/output vector index not implemented!!!')
                        vec_ix+=1
                    if report:
                        forcelog(str(self.name)+' wrote '+str(val[0])+' to '+str(module)+'.'+str(feat)+'[0:'+str(vec_ix)+']')
                else: #if not vector in ftype
                    if not override:
                        if release:
                            self.mod_feat_func[ftype][4](module, feat, False)
                        self.mod_feat_func[ftype][1](module, feat, val)
                    else:
                        self.mod_feat_func[ftype][3](module, feat, val)
                        if release:
                            self.mod_feat_func[ftype][4](module, feat, False)
                        else:
                            self.mod_feat_func[ftype][4](module, feat, True)
                    if report:
                        forcelog(str(self.name)+' wrote '+str(val)+' to '+str(module)+'.'+str(feat))
            else: #if single_vec_ix
                if not override:
                    if release:
                        forcelog('(4) release of override on single input/output vector index not implemented!!!')
                    self.mod_feat_func[ftype][2](module, feat, vec_ix, val)
                else:
                    if release:
                        forcelog('(5) release of override on single input/output vector index not implemented!!!')
                    else:
                        forcelog('(6) override single input/output vector index not implemented!!!')
                if report:
                    forcelog(str(self.name)+' wrote '+str(val)+' to '+str(module)+'.'+str(feat)+'['+str(vec_ix)+']')
        
        return
    
#setOverrideEnabled_OutputSignal(String module, String signal, boolean enabled)
#setOverrideValue_OutputSignal(String module, String signal, Object value)
        

    def connectToJVM(self, sim_path, user_dir, web_server_port, python_port):
        #Need to establish a connection to the JVM. That will happen only when the Cybersea Simulator is opened.
        #subprocess.Popen(["C:\\Users\\jonom\\_Work\\CyberSea\\revoltsim\\bin\\revoltsim64.exe", "--webServerPort="+str(web_server_port), "--pythonPort="+str(python_port), "--userdir", user_dir+str(web_server_port)])
        #log('Waiting time for the application to open...')
        #time.sleep(15)
        self.gateway = JavaGateway(gateway_parameters=GatewayParameters(port=python_port))        
        self.simulator = self.gateway.entry_point
        
        isPythonListenerStarted = None
        num_retries_connect_current = 0
        num_retries_connect_max = 100
        while isPythonListenerStarted is None and num_retries_connect_current < num_retries_connect_max:
            try:
                #check if the simulator already runs
                isPythonListenerStarted = self.simulator.isPythonListenerStarted()
            except:
                #start Cybersea if no reponse
                if num_retries_connect_current < 1:
                    subprocess.Popen([sim_path, "--webServerPort="+str(web_server_port), "--pythonPort="+str(python_port), "--userdir", user_dir+str(web_server_port)])
                    forcelog('Waiting for the CS sim to open...')
                    time.sleep(12)
                #wait 3 seconds before trying again
                num_retries_connect_current += 1
                forcelog("CS sim not started...attempt #" + str(num_retries_connect_current))
                time.sleep(3)
        
        log("CS sim started. JVM accepted python connection.")  
    

    def get_mod_feat_funcs(self):
        funcs = {}
        for feat_type in self.MODULE_FEAT_TYPES:
            if feat_type == self.SCAL_INP:
                funcs[feat_type] = [self.simulator.getScalarInputSignal, self.simulator.setScalarInputSignal, None, \
                     self.simulator.setOverrideValue_InputSignal, self.simulator.setOverrideEnabled_InputSignal]
            elif feat_type == self.SCAL_OUT:
                funcs[feat_type] = [self.simulator.getScalarOutputSignal, None, None, \
                     self.simulator.setOverrideValue_OutputSignal, self.simulator.setOverrideEnabled_OutputSignal]
            elif feat_type == self.VEC_INP:
                funcs[feat_type] = [self.simulator.getVectorInputSignal, self.simulator.setVectorInputSignal, self.simulator.setVectorInputSignalAt, \
                     self.simulator.setOverrideValue_InputSignal, self.simulator.setOverrideEnabled_InputSignal]
            elif feat_type == self.VEC_OUT:
                funcs[feat_type] = [self.simulator.getVectorOutputSignal, None, None, \
                     self.simulator.setOverrideValue_OutputSignal, self.simulator.setOverrideEnabled_OutputSignal]
            elif feat_type == self.SCAL_PARA:
                funcs[feat_type] = [self.simulator.getScalarParameter, self.simulator.setScalarParameter, None, ]
            elif feat_type == self.VEC_PARA:
                funcs[feat_type] = [self.simulator.getVectorParameter, self.simulator.setVectorParameter, self.simulator.setVectorParameterAt]
            else:
                funcs[feat_type] = None
        return funcs
            
    def load_config(self):
        cf = self.cfg_path
        if self.cfg_path is not None:
            if not ':\\' in self.cfg_path:
                self.make_abs_cfg_path()
            if self.load_cfg:
                self.simulator.configure(self.cfg_path)
            else:
                config = self.get_config()
                if len(config) < 1:
                    self.simulator.configure(self.cfg_path)
                else:
                    log('skipped load config...')
        else:
            forcelog('skipped load config - no config path!!')

        config = self.get_config()
        if len(config) < 1:
            forcelog('Simulator has empty config!!')
            
        return config
       
    def make_abs_cfg_path(self):
        relpth = self.cfg_path
        #get the current full path        
        path = os.getcwd()
        
        loops = 0
        while loops < 10:
            if '..\\' in relpth:
                relpth = relpth[3:]
                ix = path.rfind('\\')
                if ix > -1:
                    path = path[:ix]
                else:
                    forcelog('Invalid relative config path')
                    return
            else:
                break
            loops += 1
        self.cfg_path = path + '\\' + relpth
        
        
        
    def set_all_reset(self, val):
        if len(self.config) == 0:
            forcelog('No content in config!')
            return False
        log('reset all = '+str(val))
        feat = 'StateReset'
        for module in self.config.keys():
            s=str(self.config[module])
            ix0 = 0
            while True:
                ix1 = s.find(feat,ix0)
                if ix1 < 0:
                    break
                ix2 = s.find("'",ix1+1)
                if ix2 < 0:
                    print('Wrongdoings in set all reset')
                self.val(module, s[ix1:ix2], val, report = False)
                ix0 = ix2+1 
        
    def setRealTimeMode(self, mode=True):
        self.simulator.setRealtimeSimulation(mode)
        
    def runScripts(self):
        self.simulator.runAllScripts("AUTO")
        forcelog(str(self.name)+' all scripts run')
        
    def step(self, steps=1):
        if steps > 1:
            self.simulator.step(steps)
        elif steps == 1:
            self.simulator.step()
        else:
            forcelog('Invalid stepping!! ' + str(steps))
               
    def get_mod_vals(self, module, ftype=None, N=-1):
        out = {}
        cfg = self.config.get(module, None)
        if cfg is None:
            forcelog('get_mod_vals: invalid module name '+module)
            return None
        if ftype is None:
            for ft in cfg.keys():
                out[ft] = self.get_mod_vals(module, ft, N)
            return out
        else:
            if ftype in self.MODULE_FEAT_TYPES:
                feats = cfg[ftype]
                for feat in feats:
                    val0 = self.val(module, feat)
                    if 'vector' in ftype:
                        lenval0 = len(val0)
                        if N<0:
                            N=lenval0
                        val = []
                        for ix in range(N):
                            val.append(val0[ix])
                    else:
                        val = val0
                    out[feat] = val
                return out
            else:
                forcelog('get_mod_vals: invalid ftype name '+ftype)
                return {}
            
    def get_mod_feat_type(self, module, feat):
        if "__v__" in feat:
            feat = feat.split("__v__")[0]
            #log("Handling with vectors: " + feat)
        
        if module not in self.config:
            forcelog('Unknown module !! ' + module + ' -> ' + str(self.config.keys()))
            return None
        for ftyp in self.config[module].keys():
            for ft in self.config[module][ftyp]:
                if ft == feat:
                    return ftyp
        return None
            
    def get_config(self):
        config = {}
        jvm_modules = self.simulator.getModules()
        for m in jvm_modules:
            config[m] = self.get_module(m)
        return config
    
    def get_module(self, module):
        mod_feats = {}
        for ft in self.MODULE_FEAT_TYPES:
            mod_feats[ft] = self.get_module_feats(module, ft)
            if mod_feats[ft] == []:
                mod_feats.pop(ft)
        return mod_feats
         
    def get_module_feats(self, module, feat_type):

        if feat_type == self.SCAL_INP:
            jvm_feats = self.simulator.getScalarInputSignals(module)
        elif feat_type == self.SCAL_OUT:
            jvm_feats = self.simulator.getScalarOutputSignals(module)
        elif feat_type == self.VEC_INP:
            jvm_feats = self.simulator.getVectorInputSignals(module)
        elif feat_type == self.VEC_OUT:
            jvm_feats = self.simulator.getVectorOutputSignals(module)
        elif feat_type == self.SCAL_PARA:
            jvm_feats = self.simulator.getScalarParameters(module)
        elif feat_type == self.VEC_PARA:
            jvm_feats = self.simulator.getVectorParameters(module)

        feats = []
        for feat in jvm_feats:
            feats.append(feat)
        return feats


#test how to use Digitwin:

if __name__ == "__main__":

    #Cybersea app need to be manually started in advance of python run
    
    print("Connecting to CyberSea simulator and loading a configuration")    

    #Make an instance of the Digitwin class to configure the Cybersea simulator and to manipulate it
    #input the path to the config folder
    #TODO: input the path to the Cybersea app to start it atuomatically
    digitwin = DigiTwin(r'''C:\Users\jonom\_Work\CyberSea\revoltsim\configurationDRL''')
    #digitwin = DigiTwin(r'''C:\Users\jonom\_Work\CyberSea\cybersea-testapp\house''')
    
    print("Connected to CyberSea simulator and configuration loaded")    

    #set initial inputs and parameters before the simulation starts
#    digitwin.val('<module>', '<feature>', <val>)

    #simulation starts
    
    #simulation step 1:
    
    #set the reset signal high for all relevant modules, then step the sim one step, then set resets low
#    digitwin.val('<module>', 'StateResetOn', 1)
#    digitwin.step()
#    digitwin.val('<module>', 'StateResetOn', 0)

    #simulation step 2->
    steps = 25000
    i=1
    while True:
        #set inputs and parameters for next step
#        digitwin.val('<module>', '<feature>', <val>)
#        digitwin.val('<module>', '<feature>', <val>)
        digitwin.val('THR1', 'ThrustOrTorqueCmdDp', i/250)
        
        #step the simulation 1 step or X steps digitwin.step(x)
        digitwin.step()
        
        #read out any relevant output
#       val = digitwin.val('<module>', '<feature>')
        
        #input own code to manipulate sim for each step
        i+=1
        if i> steps:
            break
    
    