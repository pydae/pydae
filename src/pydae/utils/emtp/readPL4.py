import mmap
import struct
import pandas as pd
import numpy as np

class readPL4:
      
    def __init__(self, pl4file):
        self.pl4file = pl4file        
        self.miscData = {
            'deltat':0.0,
            'nvar':0,
            'pl4size':0,
            'steps':0,
            'tmax':0.0
        }

        self.types_dict = {4:'V-node', 7:'E-bran', 8:'V-bran', 9:'I-bran'}
        self.read_data()


    def read_data(self):

        miscData = self.miscData
        # open binary file for reading
        with open(self.pl4file, 'rb') as f:
            pl4 = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                
            # read DELTAT
            miscData['deltat'] = struct.unpack('<f', pl4[40:44])[0]

            # read number of vars
            miscData['nvar'] = struct.unpack('<L', pl4[48:52])[0] // 2

            # read PL4 disk size
            miscData['pl4size'] = struct.unpack('<L', pl4[56:60])[0]-1

            # compute the number of simulation miscData['steps'] from the PL4's file size
            miscData['steps'] = (miscData['pl4size'] - 5*16 - miscData['nvar']*16) // \
                                ((miscData['nvar']+1)*4)
            miscData['tmax'] = (miscData['steps']-1)*miscData['deltat']

            TYPEs = []
            TYPE_names = []
            FROMs = []
            TOs = []

            for i in range(0,miscData['nvar']):
                pos = 5*16 + i*16
                h = struct.unpack('3x1c6s6s',pl4[pos:pos+16])
                TYPEs += [int(h[0])]
                FROMs += [h[1].decode('utf-8').replace(' ','')]
                TOs += [h[2].decode('utf-8').replace(' ','')]

                if int(h[0]) in self.types_dict:
                    TYPE_names += [self.types_dict[int(h[0])]]
                else:
                    TYPE_names += [str(int(h[0]))]
                

            # generate pandas dataframe	to store the PL4's header
            self.dfHEAD = pd.DataFrame({'TYPE':TYPE_names,'FROM':FROMs,'TO':TOs})

            expsize = (5 + miscData['nvar'])*16 + miscData['steps']*(miscData['nvar']+1)*4
            nullbytes = 0
            if miscData['pl4size'] > expsize: 
                nullbytes = miscData['pl4size']-expsize

            data = np.memmap(f,dtype=np.float32,mode='r',shape=(miscData['steps'],miscData['nvar']+1),offset=(5 + miscData['nvar'])*16 + nullbytes)
            self.data = np.array(data)

            self.Time = self.data[:,0] 


    # Get desired variable data
    def get_values(self,Type,From,To):
        
        # Search for desired data in header
        df = self.dfHEAD[(self.dfHEAD['TYPE'] == Type) & (self.dfHEAD['FROM'] == From) & (self.dfHEAD['TO'] == To)]
                    
        if not df.empty:
            data_sel = self.data[:,df.index.values[0]+1] # One column shift given time vector
            
        else:
            print("Variable %s-%s of %s not found!"%(From,To,Type))
            return(None)

        return(data_sel)    


