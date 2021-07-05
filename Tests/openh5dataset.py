# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 14:53:20 2021

Functions to open PROCHIP 2-beam chip dataset
"""
import h5py
import numpy as np

def get_datasets_index_by_name(fname, match="/t0000/"):
        f = h5py.File(fname,'r')
        names,shapes,found = _get_h5_datasets(f, name=[], shape=[], found=0)    
        assert found > 0, "Specified h5 file does not exsist or have no datasets"    
        
        index_list = []
        names_list = []
        for idx,name in enumerate(names):
            if match in name:
                index_list.append(idx)
                names_list.append(name)
        f.close()        
        return index_list, names_list
        


def get_multiple_h5_datasets(fname, idx_list):
        """
        Finds datasets in HDF5 file.
        Returns the datasets specified by the dataset_index in a 16bit, n-dimensional numpy array
        If the size of the first dimension of the stack is different, between the datasets, the minimum size is choosen 
        """
        f = h5py.File(fname,'r')
        names,shapes,found = _get_h5_datasets(f, name=[], shape=[], found=0)    
        assert found > 0, "Specified h5 file does not exsist or have no datasets"    
        assert max(idx_list) < found, "Specified h5 file have less datasets than requested"    
        
        data_shape = shapes[idx_list[0]] 
        size0 = data_shape[0]
        for idx in idx_list[1::]:
             size0 = min(size0,shapes[idx][0])   
        data = np.zeros([len(idx_list), size0, *data_shape[1::]])
        for key,idx in enumerate(idx_list):
            stack = np.single(f[names[idx]])
            data [key,...] = stack[0:size0,...]
        f.close()
        return data, found
    
def _get_h5_datasets(g, name, shape, found) :
        """
        Extracts the dataset location (and its shape).
        It is operated recursively in the h5 file.
        """
       
        if isinstance(g,h5py.Dataset):   
            found += 1
            name.append(g.name)
            shape.append(g.shape)
            
        if isinstance(g, h5py.File) or isinstance(g, h5py.Group):
            for key,val in dict(g).items() :
                
                name,shape,found = _get_h5_datasets(val,name,shape,found)
                 
        return name,shape,found 
    
    
if __name__ == '__main__':
    """
    For the data sent on 18/06/2021, the best datasets to look at, are indicated in the filename. They are:
     
    210506_143951_PROCHIP_multichannel_ROI_chip0_dataset65 : dataset 65
    210528_194608_PROCHIP_multichannel_ROI_dataset1 : dataset 1
    210528_195219_PROCHIP_multichannel_ROI_dataset11: dataset 11
    210611_150316_PROCHIP_SIM_ROI_dataset9 : dataset 9
    210611_152833_PROCHIP_SIM_ROI_dataset51 : dataset 51
    210611_131018_PROCHIP_SIM_rodamine :dataset 0
    """

    dataset_num = 11
    file_name = '210528_195219_PROCHIP_multichannel_ROI_dataset11.h5'

    
    t_idx = f'/t{dataset_num:04d}/'    
    index_list, index_names = get_datasets_index_by_name(file_name, t_idx)
    data, _ =  get_multiple_h5_datasets(file_name, index_list)
    
    
    print(data.shape)







    
    
