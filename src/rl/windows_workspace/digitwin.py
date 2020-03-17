# -*- coding: utf-8 -*-
"""
Created on Mon May  7 14:11:34 2018

@author: jonom
"""

from py4j.java_gateway import JavaGateway, Py4JNetworkError
from py4j.java_gateway import GatewayParameters
import subprocess
import time
from utils.log import log, forcelog

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
    
    def __init__(self, name, load_cfg, sim_path, cfg_path, python_port):
        self.name = name


        log('Opening simulator')
        self.load_cfg = self.connectToJVM(sim_path, python_port)
        if load_cfg:
            self.load_cfg = True

        self.mod_feat_func = self.get_mod_feat_funcs()
        self.cfg_path = cfg_path

        # load config
        log("CS sim loading config.")    
        loaded_cfg = self.load_config()
        log("CS sim loaded config.")    

        # read config
        self.config = self.get_config()
        log("CS sim config read.")    
        
#        if not loaded_cfg:
#            self.set_all_reset(1)
#            self.step(50)

        # step sim as fast as possible
        self.setRealTimeMode(False)
    
####
#method to read or write values to module feature (feat) parameters in the digital twin (simulator)
#no val indicates a read request
#report flag indicates if written val should be printed to std out.
#some features are vectors.  if val = list, len(val) indices of this vector will be written
# if val = number, all indices of this vecor will be written the same value.
    def val(self, module, feat, val=None, report=True):
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
        if val is None:
            val = self.mod_feat_func[ftype][0](module, feat)
            if report:
                log(str(self.name)+' read '+str(val)+' from '+str(module)+'.'+str(feat))
            return val
        else:
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
                        self.mod_feat_func[ftype][2](module, feat, vec_ix, float(v))
#                        if report:
#                            forcelog(str(self.name)+' wrote '+str(v)+' to '+str(module)+'.'+str(feat)+'['+str(vec_ix)+']')
                        vec_ix+=1
                    if report:
                        forcelog(str(self.name)+' wrote '+str(val[0])+' to '+str(module)+'.'+str(feat)+'[0:'+str(vec_ix)+']')
                else: #if not vector in ftype
                    self.mod_feat_func[ftype][1](module, feat, float(val))
                    if report:
                        forcelog(str(self.name)+' wrote '+str(val)+' to '+str(module)+'.'+str(feat))
            else: #if single_vec_ix
                self.mod_feat_func[ftype][2](module, feat, vec_ix, float(val))
                if report:
                    forcelog(str(self.name)+' wrote '+str(val)+' to '+str(module)+'.'+str(feat)+'['+str(vec_ix)+']')
                
            #forcelog(str(self.name)+' wrote '+str(val)+' to '+str(module)+'.'+str(feat))
        

    def connectToJVM(self, sim_path, python_port):
        #Need to establish a connection to the JVM. That will happen only when the Cybersea Simulator is opened.
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
                log("CS sim already started")
                return False
            except:
                #start Cybersea if no reponse
                if num_retries_connect_current < 1:
                    subprocess.Popen([sim_path, "--pythonPort="+str(python_port)])
                    forcelog('Waiting for the CS sim to open...')
                    time.sleep(12)
                #wait 3 seconds before trying again
                num_retries_connect_current += 1
                forcelog("CS sim not started...attempt #" + str(num_retries_connect_current))
                time.sleep(3)
        
                log("CS sim started. JVM accepted python connection.")
                return True
    
    def get_mod_feat_funcs(self):
        funcs = {}
        for feat_type in self.MODULE_FEAT_TYPES:
            if feat_type == self.SCAL_INP:
                funcs[feat_type] = [self.simulator.getScalarInputSignal, self.simulator.setScalarInputSignal]
            elif feat_type == self.SCAL_OUT:
                funcs[feat_type] = [self.simulator.getScalarOutputSignal, self.simulator.setScalarOutputSignal]
            elif feat_type == self.VEC_INP:
                funcs[feat_type] = [self.simulator.getVectorInputSignal, self.simulator.setVectorInputSignal, self.simulator.setVectorInputSignalAt]
            elif feat_type == self.VEC_OUT:
                funcs[feat_type] = [self.simulator.getVectorOutputSignal, self.simulator.setVectorOutputSignal]
            elif feat_type == self.SCAL_PARA:
                funcs[feat_type] = [self.simulator.getScalarParameter, self.simulator.setScalarParameter]
            elif feat_type == self.VEC_PARA:
                funcs[feat_type] = [self.simulator.getVectorParameter, self.simulator.setVectorParameter, self.simulator.setVectorParameterAt]
            else:
                funcs[feat_type] = None
        return funcs
            
    def load_config(self):
        if self.cfg_path is not None and self.load_cfg:
            self.simulator.configure(self.cfg_path)
            return True
        else:
            log('skipped config...')
            return False
       
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
                self.val(module, s[ix1:ix2], val, False)
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
  
