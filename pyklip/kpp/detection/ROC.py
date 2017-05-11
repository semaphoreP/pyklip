__author__ = 'jruffio'
import os
import astropy.io.fits as pyfits
from glob import glob
import multiprocessing as mp
import numpy as np
from scipy.signal import convolve2d

from pyklip.kpp.utils.kppSuperClass import KPPSuperClass
from pyklip.kpp.stat.stat_utils import *
from pyklip.kpp.utils.GOI import *
import pyklip.kpp.utils.mathfunc as kppmath
import pyklip.kpp.utils.GPIimage as gpiim

class ROC(KPPSuperClass):
    """
    Class for SNR calculation.
    """
    def __init__(self,filename,filename_detec,
                 inputDir = None,
                 outputDir = None,
                 mute=None,
                 N_threads=None,
                 label = None,
                 detec_distance = None,
                 ignore_distance = None,
                 GOI_list_folder = None,
                 threshold_sampling = None,
                 overwrite = False,
                 IWA = None,
                 OWA = None):
        """


        :param filename: Filename of the file on which to calculate the metric. It should be the complete path unless
                        inputDir is defined.
                        It can include wild characters. The file will be selected using the first output of glob.glob().
        :param filename_noSignal: One should be careful with this one since it requires it should find the same number
                            of files with no signal than normal images when calling glob.glob().
                            Besides one has to check that the ordering of glob outputs are matching for both lists.
        :param mute: If True prevent printed log outputs.
        :param N_threads: Number of threads to be used for the metrics and the probability calculations.
                        If None use mp.cpu_count().
                        If -1 do it sequentially.
                        Note that it is not used for this super class.
        :param label: Define the suffix to the output folder when it is not defined. cf outputDir. Default is "default".
        """
        # allocate super class
        super(ROC, self).__init__(filename,
                                     inputDir = inputDir,
                                     outputDir = outputDir,
                                     folderName = None,
                                     mute=mute,
                                     N_threads=N_threads,
                                     label=label,
                                     overwrite = overwrite)

        if detec_distance is None:
            self.detec_distance = 2
        else:
            self.detec_distance = detec_distance

        if ignore_distance is None:
            self.ignore_distance = 10
        else:
            self.ignore_distance = ignore_distance

        if threshold_sampling is None:
            self.threshold_sampling = np.linspace(0.0,20,200)
        else:
            self.threshold_sampling = threshold_sampling

        self.filename_detec = filename_detec
        self.GOI_list_folder = GOI_list_folder

        self.IWA = IWA
        self.OWA = OWA



    def initialize(self,inputDir = None,
                         outputDir = None,
                         folderName = None,
                         compact_date = None,
                         label = None):
        """
        Initialize the non general inputs that are needed for the metric calculation and load required files.

        For this super class it simply reads the input file including fits headers and store it in self.image.
        One can also overwrite inputDir, outputDir which is basically the point of this function.
        The file is assumed here to be a fits containing a 2D image or a GPI 3D cube (assumes 37 spectral slice).

        Example for inherited classes:
        It can read the PSF cube or define the hat function.
        It can also read the template spectrum in a 3D scenario.
        It could also overwrite this function in case it needs to read multiple files or non fits file.

        :param inputDir: If defined it allows filename to not include the whole path and just the filename.
                        Files will be read from inputDir.
                        Note tat inputDir might be redefined using initialize at any point.
                        If inputDir is None then filename is assumed to have the absolute path.
        :param outputDir: Directory where to create the folder containing the outputs.
                        Note tat inputDir might be redefined using initialize at any point.
                        If outputDir is None:
                            If inputDir is defined: outputDir = inputDir+os.path.sep+"planet_detec_"
        :param folderName: Name of the folder containing the outputs. It will be located in outputDir.
                        Default folder name is "default_out".
                        The convention is to have one folder per spectral template.
                        If the keyword METFOLDN is available in the fits file header then the keyword value is used no
                        matter the input.
        :param label: Define the suffix to the output folder when it is not defined. cf outputDir. Default is "default".

        :return: None
        """
        if not self.mute:
            print("~~ INITializing "+self.__class__.__name__+" ~~")

        # The super class already read the fits file
        init_out = super(ROC, self).initialize(inputDir = inputDir,
                                         outputDir = outputDir,
                                         folderName = folderName,
                                         label=label)


        # Get center of the image (star position)
        try:
            # Retrieve the center of the image from the fits headers.
            self.center = [self.exthdr['PSFCENTX'], self.exthdr['PSFCENTY']]
        except:
            # If the keywords could not be found the center is defined as the middle of the image
            if not self.mute:
                print("Couldn't find PSFCENTX and PSFCENTY keywords.")
            self.center = [(self.nx-1)/2,(self.ny-1)/2]

        if self.label == "CADI":
            self.center = [140,140]

        try:
            self.folderName = self.exthdr["METFOLDN"]
        except:
            pass



        # Check file existence and define filename_path
        if self.inputDir is None or os.path.isabs(self.filename_detec):
            try:
                self.filename_detec_path = os.path.abspath(glob(self.filename_detec)[self.id_matching_file])
                self.N_matching_files = len(glob(self.filename_detec))
            except:
                raise Exception("File "+self.filename_detec+"doesn't exist.")
        else:
            try:
                self.filename_detec_path = os.path.abspath(glob(self.inputDir+os.path.sep+self.filename_detec)[self.id_matching_file])
                self.N_matching_files = len(glob(self.inputDir+os.path.sep+self.filename_detec))
            except:
                raise Exception("File "+self.inputDir+os.path.sep+self.filename_detec+" doesn't exist.")

        with open(self.filename_detec_path, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            csv_as_list = list(reader)
            self.detec_table_labels = csv_as_list[0]
            self.detec_table = np.array(csv_as_list[1::], dtype='string').astype(np.float)
        if not self.mute:
            print("Opened: "+self.filename_detec_path)


        self.N_detec = self.detec_table.shape[0]
        self.val_id = self.detec_table_labels.index("value")
        self.x_id = self.detec_table_labels.index("x")
        self.y_id = self.detec_table_labels.index("y")

        file_ext_ind = os.path.basename(self.filename_detec_path)[::-1].find(".")
        self.prefix = os.path.basename(self.filename_detec_path)[:-(file_ext_ind+1)]
        self.suffix = "ROC"
        return init_out

    def check_existence(self):
        """

        :return: False
        """

        file_exist = (len(glob(self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+self.suffix+'.csv')) >= 1)

        if file_exist and not self.mute:
            print("Output already exist: "+self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+self.suffix+'.csv')

        if self.overwrite and not self.mute:
            print("Overwriting is turned ON!")

        return file_exist and not self.overwrite


    def calculate(self):
        """

        :param N: Defines the width of the ring by the number of pixels it has to contain
        :return: self.image the imput fits file.
        """
        if not self.mute:
            print("~~ Calculating "+self.__class__.__name__+" with parameters " + self.suffix+" ~~")

        if self.GOI_list_folder is not None:
            x_real_object_list,y_real_object_list = get_pos_known_objects(self.prihdr,self.exthdr,self.GOI_list_folder,xy = True,ignore_fakes=True,IWA=self.IWA,OWA=self.OWA)

        row_object_list,col_object_list = get_pos_known_objects(self.prihdr,self.exthdr,IWA=self.IWA,OWA=self.OWA)

        self.false_detec_proba_vec = []
        # self.true_detec_proba_vec = []
        # print(x_real_object_list,y_real_object_list)
        # Loop over all the local maxima stored in the detec csv file
        for k in range(self.N_detec):
            val_criter = self.detec_table[k,self.val_id]
            x_pos = self.detec_table[k,self.x_id]
            y_pos = self.detec_table[k,self.y_id]

            #remove the detection if it is a real object
            if self.GOI_list_folder is not None:
                reject = False
                for x_real_object,y_real_object  in zip(x_real_object_list,y_real_object_list):
                    #print(np.sqrt((x_pos-x_real_object)**2+(y_pos-y_real_object)**2 ),self.detec_distance**2,self.ignore_distance**2)
                    if (x_pos-x_real_object)**2+(y_pos-y_real_object)**2 < self.ignore_distance**2:
                        reject = True
                        break

                # too_close = False
                # for x_real_object,y_real_object  in zip(x_real_object_list,y_real_object_list):
                #     #print(np.sqrt((x_pos-x_real_object)**2+(y_pos-y_real_object)**2 ),self.detec_distance**2,self.ignore_distance**2)
                #     if (x_pos-x_real_object)**2+(y_pos-y_real_object)**2 < self.detec_distance**2:
                #         too_close = True
                #         # self.true_detec_proba_vec.append(val_criter)
                #         if not self.mute:
                #             print("Real object detected.")
                #         break
                #     elif (x_pos-x_real_object)**2+(y_pos-y_real_object)**2 < self.ignore_distance**2:
                #         too_close = True
                #         if not self.mute:
                #             print("Local maxima ignored. Too close to known object")
                #         break
            if reject:
                continue

            if self.IWA is not None:
                if np.sqrt( (x_pos)**2+(y_pos)**2) < self.IWA:
                    continue
            if self.OWA is not None:
                if np.sqrt( (x_pos)**2+(y_pos)**2) > self.OWA:
                    continue

            self.false_detec_proba_vec.append(val_criter)

        self.true_detec_proba_vec = [self.image[np.round(row_real_object),np.round(col_real_object)] \
                                     for row_real_object,col_real_object in zip(row_object_list,col_object_list)]
        self.true_detec_proba_vec = np.array(self.true_detec_proba_vec)[np.where(~np.isnan(self.true_detec_proba_vec))]
        # print(self.false_detec_proba_vec)
        # print(self.true_detec_proba_vec)

        self.N_false_pos = np.zeros(self.threshold_sampling.shape)
        self.N_true_detec = np.zeros(self.threshold_sampling.shape)
        for id,threshold_it in enumerate(self.threshold_sampling):
            self.N_false_pos[id] = np.sum(self.false_detec_proba_vec >= threshold_it)
            self.N_true_detec[id] = np.sum(self.true_detec_proba_vec >= threshold_it)

        # import matplotlib.pyplot as plt
        # plt.figure(1)
        # plt.plot(self.N_false_pos,self.N_true_detec)
        # plt.show()

        return zip(self.threshold_sampling,self.N_false_pos,self.N_true_detec)


    def save(self):
        """

        :return: None
        """

        if not os.path.exists(self.outputDir+os.path.sep+self.folderName):
            os.makedirs(self.outputDir+os.path.sep+self.folderName)

        if not self.mute:
            print("Saving: "+self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+self.suffix+'.csv')
        with open(self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+self.suffix+'.csv', 'w+') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=';')
            csvwriter.writerows([["value","N false pos","N true pos"]])
            csvwriter.writerows(zip(self.threshold_sampling,self.N_false_pos,self.N_true_detec))
        return None

    def load(self):
        """

        :return: None
        """

        return None

def gather_ROC(filename_filter,mute = False):
    """
    Build the combined ROC curve from individual frame ROC curve.
    It looks for all the file matching filename_filter using glob.glob and then add each individual ROC to build the
    master ROC.

    Plot master_N_false_pos vs master_N_true_detec to get a ROC curve.

    :param filename_filter: Filename filter with wild characters indicating which files to pick
    :param mute: If True, mute prints. Default is False.
    :return: threshold_sampling,master_N_false_pos,master_N_true_detec:
        threshold_sampling: The metric sampling. It is the curve parametrization.
        master_N_false_pos: Number of false positives as a function of threshold_sampling
        master_N_true_detec: Number of true positives as a function of threshold_sampling
    """
    file_list = glob(filename_filter)

    with open(file_list[0], 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        csv_as_list = list(reader)
        detec_table_labels = csv_as_list[0]
        detec_table = np.array(csv_as_list[1::], dtype='string').astype(np.float)

    threshold_sampling = detec_table[:,0]
    master_N_false_pos = detec_table[:,1]
    master_N_true_detec = detec_table[:,2]

    for filename in file_list[1::]:
        with open(filename, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            csv_as_list = list(reader)
            detec_table_labels = csv_as_list[0]
            detec_table = np.array(csv_as_list[1::], dtype='string').astype(np.float)
        if not mute:
            print("Opened: "+filename)

        master_N_false_pos = master_N_false_pos+detec_table[:,1]
        master_N_true_detec = master_N_true_detec+detec_table[:,2]

    return threshold_sampling,master_N_false_pos,master_N_true_detec


def gather_multiple_ROCs(base_dir,filename_filter_list,mute = False,epoch_suffix=None,stars2ignore=None,band = None):
    """
    Build the multiple combined ROC curve from individual frame ROC curve while making sure they have the same inputs.
    If the folders are organized following the convention below then it will make sure there is a ROC file for each
    filename_filter in each epoch. Otherwise it skips the epoch.

    The folders need to be organized as:
     base_dir/TARGET/autoreduced/EPOCH_Spec/filename_filter

    In the function TARGET and EPOCH are wild characters.

    It looks for all the file matching filename_filter using glob.glob and then add each individual ROC to build the
    master ROC.

    Plot master_N_false_pos vs master_N_true_detec to get a ROC curve.

    :param base_dir: Base directory from which the file search go.
    :param filename_filter: Filename filter with wild characters indicating which files to pick.
    :param mute: If True, mute prints. Default is False.
    :return: threshold_sampling,master_N_false_pos,master_N_true_detec:
        threshold_sampling: The metric sampling. It is the curve parametrization.
        master_N_false_pos: Number of false positives as a function of threshold_sampling
        master_N_true_detec: Number of true positives as a function of threshold_sampling
    """

    N_ROC = len(filename_filter_list)
    threshold_sampling_list = [[]]*N_ROC
    master_N_false_pos_list = [[]]*N_ROC
    master_N_true_detec_list = [[]]*N_ROC

    if epoch_suffix is None:
        epoch_suffix = ""

    if stars2ignore is None:
        stars2ignore=[]


    if band is None:
        band = "H"

    dirs_to_reduce = os.listdir(base_dir)
    N=0
    for star_name in dirs_to_reduce:
        if not star_name.startswith('.') and star_name not in stars2ignore:
            #print(star_name)

            epochDir_glob = glob(base_dir+star_name+os.path.sep+"autoreduced_kpop"+os.path.sep+"*_*_Spec"+epoch_suffix+os.path.sep)

            compactdate_list = [int(os.path.abspath(epochDir).split(os.path.sep)[-1].split("_")[0]) for epochDir in epochDir_glob]
            N_cubes_list = [len(glob(epochDir.replace("autoreduced_kpop","autoreduced")+os.path.sep+"S*_spdc_distorcorr.fits")) for epochDir in epochDir_glob]
            # Find the first valid epoch
            if len(compactdate_list)==0:
                continue
            compactdate_list,N_cubes_list = zip(*sorted(zip(compactdate_list,N_cubes_list)))
            epochDir = epochDir_glob[np.where(N_cubes_list>20)[0][0]]
            # exit()
            # for epochDir in epochDir_glob:
            if 1:
                inputDir = os.path.abspath(epochDir)
                epoch_folder_splitted = inputDir.split(os.path.sep)[-1].split("_")
                compact_date = epoch_folder_splitted[0]
                filter_name = epoch_folder_splitted[1]

                if filter_name != band:
                    continue

                file_list = []
                for filename_filter in filename_filter_list:
                    try:
                        file_list.append(glob(inputDir+os.path.sep+filename_filter)[0])
                        # if not mute:
                        #     print("ROC: {0} in {1}. Adding.".format(filename_filter,inputDir))
                    except:
                        pass
                        # if not mute:
                        #     print("ROC: {0} unvailable in {1}. Skipping".format(filename_filter,inputDir))

                if len(file_list) == N_ROC:
                    # print(file_list)
                    N=N+1
                    for index,filename in enumerate(file_list):
                        with open(filename, 'rb') as csvfile:
                            reader = csv.reader(csvfile, delimiter=';')
                            csv_as_list = list(reader)
                            detec_table_labels = csv_as_list[0]
                            detec_table = np.array(csv_as_list[1::], dtype='string').astype(np.float)

                        try:
                            master_N_false_pos_list[index] = master_N_false_pos_list[index]+detec_table[:,1]
                            master_N_true_detec_list[index] = master_N_true_detec_list[index]+detec_table[:,2]
                        except:
                            threshold_sampling_list[index] = detec_table[:,0]
                            master_N_false_pos_list[index] = detec_table[:,1]
                            master_N_true_detec_list[index] = detec_table[:,2]

    print("N files = {0}".format(N))

    return threshold_sampling_list,master_N_false_pos_list,master_N_true_detec_list,N


def get_all_false_pos(base_dir,filename_filter_list,threshold,mute = False,epoch_suffix=None,stars2ignore=None,IFSfilter=None):
    """
    Build the multiple combined ROC curve from individual frame ROC curve while making sure they have the same inputs.
    If the folders are organized following the convention below then it will make sure there is a ROC file for each
    filename_filter in each epoch. Otherwise it skips the epoch.

    The folders need to be organized as:
     base_dir/TARGET/autoreduced/EPOCH_Spec/filename_filter

    In the function TARGET and EPOCH are wild characters.

    It looks for all the file matching filename_filter using glob.glob and then add each individual ROC to build the
    master ROC.

    Plot master_N_false_pos vs master_N_true_detec to get a ROC curve.

    :param base_dir: Base directory from which the file search go.
    :param filename_filter: Filename filter with wild characters indicating which files to pick.
    :param mute: If True, mute prints. Default is False.
    :return: threshold_sampling,master_N_false_pos,master_N_true_detec:
        threshold_sampling: The metric sampling. It is the curve parametrization.
        master_N_false_pos: Number of false positives as a function of threshold_sampling
        master_N_true_detec: Number of true positives as a function of threshold_sampling
    """

    N_ROC = len(filename_filter_list)
    metric_list = [[]]*N_ROC
    pa_list = [[]]*N_ROC
    sep_list = [[]]*N_ROC

    if epoch_suffix is None:
        epoch_suffix = ""

    if stars2ignore is None:
        stars2ignore=[]

    if IFSfilter is None:
        IFSfilter="*"

    dirs_to_reduce = os.listdir(base_dir)
    # dirs_to_reduce = ["HD_202917","c_Eri"]
    N=0
    for star_name in dirs_to_reduce:
        if not star_name.startswith('.') and star_name not in stars2ignore:
            #print(star_name)

            epochDir_glob = glob(base_dir+star_name+os.path.sep+"autoreduced_kpop"+os.path.sep+"*_{0}_Spec".format(IFSfilter)+epoch_suffix+os.path.sep)

            compactdate_list = [int(os.path.abspath(epochDir).split(os.path.sep)[-1].split("_")[0]) for epochDir in epochDir_glob]
            N_cubes_list = [len(glob(epochDir.replace("autoreduced_kpop","autoreduced")+os.path.sep+"S*_spdc_distorcorr.fits")) for epochDir in epochDir_glob]
            # Find the first valid epoch
            if len(compactdate_list)==0:
                continue
            compactdate_list,N_cubes_list = zip(*sorted(zip(compactdate_list,N_cubes_list)))
            epochDir = epochDir_glob[np.where(N_cubes_list>20)[0][0]]
            # exit()
            # for epochDir in epochDir_glob:
            if 1:
                inputDir = os.path.abspath(epochDir)

                file_list = []
                for filename_filter in filename_filter_list:
                    try:
                        file_list.append(glob(inputDir+os.path.sep+filename_filter)[0])
                        # if not mute:
                        #     print("ROC: {0} in {1}. Adding.".format(filename_filter,inputDir))
                    except:
                        pass
                        # if not mute:
                        #     print("ROC: {0} unvailable in {1}. Skipping".format(filename_filter,inputDir))

                if len(file_list) == N_ROC:
                    # print(file_list)
                    N=N+1
                    for index,filename in enumerate(file_list):
                        with open(filename, 'rb') as csvfile:
                            reader = csv.reader(csvfile, delimiter=';')
                            csv_as_list = list(reader)
                            detec_table_labels = csv_as_list[0]
                            detec_table = np.array(csv_as_list[1::], dtype='string').astype(np.float)

                        metric_id = detec_table_labels.index("value")
                        pa_id = detec_table_labels.index("PA")
                        sep_id = detec_table_labels.index("Sep (as)")

                        above_thres = np.where(detec_table[:,metric_id]>threshold)

                        try:
                            metric_list[index] = metric_list[index]+detec_table[above_thres[0],metric_id].tolist()
                            pa_list[index] = pa_list[index]+detec_table[above_thres[0],pa_id].tolist()
                            sep_list[index] = sep_list[index]+detec_table[above_thres[0],sep_id].tolist()
                        except:
                            metric_list[index] = detec_table[above_thres[0],metric_id].tolist()
                            pa_list[index] = detec_table[above_thres[0],pa_id].tolist()
                            sep_list[index] = detec_table[above_thres[0],sep_id].tolist()


    print("N files = {0}".format(N))

    return metric_list,pa_list,sep_list



def get_metrics_stat(base_dir,filename_filter_list,IOWA,bins,GOI_list_folder,mute = False,epoch_suffix=None,stars2ignore=None,IFSfilter=None):
    """
    Build the multiple combined ROC curve from individual frame ROC curve while making sure they have the same inputs.
    If the folders are organized following the convention below then it will make sure there is a ROC file for each
    filename_filter in each epoch. Otherwise it skips the epoch.

    The folders need to be organized as:
     base_dir/TARGET/autoreduced/EPOCH_Spec/filename_filter

    In the function TARGET and EPOCH are wild characters.

    It looks for all the file matching filename_filter using glob.glob and then add each individual ROC to build the
    master ROC.

    Plot master_N_false_pos vs master_N_true_detec to get a ROC curve.

    :param base_dir: Base directory from which the file search go.
    :param filename_filter: Filename filter with wild characters indicating which files to pick.
    :param mute: If True, mute prints. Default is False.
    :return: threshold_sampling,master_N_false_pos,master_N_true_detec:
        threshold_sampling: The metric sampling. It is the curve parametrization.
        master_N_false_pos: Number of false positives as a function of threshold_sampling
        master_N_true_detec: Number of true positives as a function of threshold_sampling
    """

    N_ROC = len(filename_filter_list)
    hist_list = [[]]*N_ROC

    if epoch_suffix is None:
        epoch_suffix = ""

    IWA,OWA = IOWA

    if stars2ignore is None:
        stars2ignore=[]

    if IFSfilter is None:
        IFSfilter="*"

    dirs_to_reduce = os.listdir(base_dir)
    # dirs_to_reduce = ["HD_202917","c_Eri"]
    N=0
    for star_name in dirs_to_reduce:
        if not star_name.startswith('.') and star_name not in stars2ignore:
            #print(star_name)

            epochDir_glob = glob(base_dir+star_name+os.path.sep+"autoreduced_kpop"+os.path.sep+"*_{0}_Spec".format(IFSfilter)+epoch_suffix+os.path.sep)

            compactdate_list = [int(os.path.abspath(epochDir).split(os.path.sep)[-1].split("_")[0]) for epochDir in epochDir_glob]
            N_cubes_list = [len(glob(epochDir.replace("autoreduced_kpop","autoreduced")+os.path.sep+"S*_spdc_distorcorr.fits")) for epochDir in epochDir_glob]
            # Find the first valid epoch
            if len(compactdate_list)==0:
                continue
            compactdate_list,N_cubes_list = zip(*sorted(zip(compactdate_list,N_cubes_list)))
            epochDir = epochDir_glob[np.where(N_cubes_list>20)[0][0]]
            # exit()
            # for epochDir in epochDir_glob:
            if 1:
                inputDir = os.path.abspath(epochDir)

                file_list = []
                for filename_filter in filename_filter_list:
                    try:
                        file_list.append(glob(inputDir+os.path.sep+filename_filter)[0])
                        # if not mute:
                        #     print("ROC: {0} in {1}. Adding.".format(filename_filter,inputDir))
                    except:
                        pass
                        # if not mute:
                        #     print("ROC: {0} unvailable in {1}. Skipping".format(filename_filter,inputDir))

                if len(file_list) == N_ROC:
                    # print(file_list)
                    N=N+1
                    print(N)
                    for index,filename in enumerate(file_list):

                        hdulist = pyfits.open(filename)
                        exthdr = hdulist[1].header
                        prihdr = hdulist[0].header
                        image = hdulist[1].data
                        ny,nx = image.shape

                        image = mask_known_objects(image,prihdr,exthdr,GOI_list_folder, mask_radius = 7)

                        x_cen,y_cen = [exthdr['PSFCENTX'], exthdr['PSFCENTY']]

                        x_grid, y_grid = np.meshgrid(np.arange(nx)-x_cen, np.arange(ny)-y_cen)
                        # Calculate the radial distance of each pixel
                        r_grid = abs(x_grid +y_grid*1j)
                        th_grid = np.arctan2(x_grid,y_grid)

                        image_selec = np.where(np.isfinite(image)*(r_grid>gpiim.as2pix(IWA))*(r_grid<gpiim.as2pix(OWA)))
                        # print(bins)
                        # print(image[image_selec])

                        H,xedges = np.histogram(image[image_selec],bins = bins)
                        try:
                            hist_list[index] = hist_list[index]+H
                        except:
                            hist_list[index] = H


    print("N files = {0}".format(N))

    bin_center = [(r1+r2)/2. for r1,r2 in zip(bins[0:-1],bins[1:])]

    return hist_list,bin_center


def get_candidates(base_dir,filename_filter_list,threshold,mute = False,epoch_suffix=None,IWA=None,OWA=None,GOI_list_folder=None,ignore_distance=None,detec_distance=None,stars2ignore=None):
    """
    Build the multiple combined ROC curve from individual frame ROC curve while making sure they have the same inputs.
    If the folders are organized following the convention below then it will make sure there is a ROC file for each
    filename_filter in each epoch. Otherwise it skips the epoch.

    The folders need to be organized as:
     base_dir/TARGET/autoreduced/EPOCH_Spec/filename_filter

    In the function TARGET and EPOCH are wild characters.

    It looks for all the file matching filename_filter using glob.glob and then add each individual ROC to build the
    master ROC.

    Plot master_N_false_pos vs master_N_true_detec to get a ROC curve.

    :param base_dir: Base directory from which the file search go.
    :param filename_filter: Filename filter with wild characters indicating which files to pick.
    :param mute: If True, mute prints. Default is False.
    :return: threshold_sampling,master_N_false_pos,master_N_true_detec:
        threshold_sampling: The metric sampling. It is the curve parametrization.
        master_N_false_pos: Number of false positives as a function of threshold_sampling
        master_N_true_detec: Number of true positives as a function of threshold_sampling
    """

    N_ROC = len(filename_filter_list)
    metric_list = [[]]*N_ROC
    pa_list = [[]]*N_ROC
    sep_list = [[]]*N_ROC
    star_name_list = [[]]*N_ROC
    is_GOI_list = [[]]*N_ROC
    N_cubes_list = [[]]*N_ROC
    compact_date_list = [[]]*N_ROC
    filter_list = [[]]*N_ROC
    N_GOI_list = [0]*N_ROC
    N_detected_GOI_list = [0]*N_ROC
    visible_GOI_list = [[]]*N_ROC

    if epoch_suffix is None:
        epoch_suffix = ""

    if ignore_distance is None:
        ignore_distance=10

    if detec_distance is None:
        detec_distance=3

    if stars2ignore is None:
        stars2ignore=[]

    dirs_to_reduce = os.listdir(base_dir)
    # dirs_to_reduce = ["HD_202917","c_Eri"]
    # dirs_to_reduce = ["HR_2562"]
    N=0
    for star_name in dirs_to_reduce:
        if not star_name.startswith('.') and star_name not in stars2ignore:
            #print(star_name)

            epochDir_glob = glob(base_dir+star_name+os.path.sep+"autoreduced_kpop"+os.path.sep+"*_*_Spec"+epoch_suffix+os.path.sep)

            for epochDir in epochDir_glob:
                inputDir = os.path.abspath(epochDir)
                epoch_folder_splitted = inputDir.split(os.path.sep)[-1].split("_")
                compact_date = epoch_folder_splitted[0]
                filter_name = epoch_folder_splitted[1]

                file_list = []
                for filename_filter in filename_filter_list:
                    try:
                        file_list.append(glob(inputDir+os.path.sep+filename_filter)[0])
                        # if not mute:
                        #     print("ROC: {0} in {1}. Adding.".format(filename_filter,inputDir))
                    except:
                        pass
                        # if not mute:
                        #     print("ROC: {0} unvailable in {1}. Skipping".format(filename_filter,inputDir))

                if len(file_list) == N_ROC:
                    # print(file_list)
                    N=N+1
                    for index,filename in enumerate(file_list):
                        with open(filename, 'rb') as csvfile:
                            reader = csv.reader(csvfile, delimiter=';')
                            csv_as_list = list(reader)
                            detec_table_labels = csv_as_list[0]
                            detec_table = np.array(csv_as_list[1::], dtype='string').astype(np.float)

                        metric_id = detec_table_labels.index("value")
                        pa_id = detec_table_labels.index("PA")
                        sep_id = detec_table_labels.index("Sep (as)")

                        N_detec = detec_table.shape[0]
                        x_id = detec_table_labels.index("x")
                        y_id = detec_table_labels.index("y")

                        detec_in_range = np.ones(N_detec)
                        if IWA is not None:
                            detec_in_range = detec_in_range*np.array(detec_table[:,sep_id]>IWA)
                        if IWA is not None:
                            detec_in_range = detec_in_range*np.array(detec_table[:,sep_id]<OWA)

                        valid_detec = np.where((detec_table[:,metric_id]>threshold)*detec_in_range)

                        # isgoi = 1 if detection less than 1 pixel away from GOI
                        # isgoi = -1 if detection less than 10 pixel away from GOI
                        # isgoi = 0 otherwise
                        is_GOI = np.zeros(N_detec)

                        filename_fits = filename.split("-DetecTh")[0]+".fits"
                        hdulist = pyfits.open(filename_fits)
                        exthdr = hdulist[1].header
                        prihdr = hdulist[0].header

                        N_cubes = 0
                        EOList = False
                        while not EOList:
                            try:
                                prihdr["FILE_{0}".format(N_cubes)]
                            except:
                                EOList = True
                            N_cubes = N_cubes +1


                        if GOI_list_folder is not None:
                            x_real_object_list,y_real_object_list = get_pos_known_objects(prihdr,exthdr,GOI_list_folder,xy = True,ignore_fakes=True,IWA=gpiim.as2pix(IWA),OWA=gpiim.as2pix(OWA))
                            x_real_object_list = np.array(x_real_object_list)
                            y_real_object_list = np.array(y_real_object_list)
                            sep_real_object_list,pa_real_object_list = get_pos_known_objects(prihdr,exthdr,GOI_list_folder,pa_sep=True,ignore_fakes=True,IWA=gpiim.as2pix(IWA),OWA=gpiim.as2pix(OWA))
                            sep_real_object_list = np.array(sep_real_object_list)
                            pa_real_object_list = np.array(pa_real_object_list)

                            GOI_in_range = np.ones(len(sep_real_object_list))
                            if IWA is not None:
                                GOI_in_range = GOI_in_range*(np.array(sep_real_object_list)>IWA)
                            if OWA is not None:
                                GOI_in_range = GOI_in_range*(np.array(sep_real_object_list)<OWA)
                            GOI_in_range = np.where(GOI_in_range)
                            # N_GOI_list[index] = N_GOI_list[index] + np.nansum(GOI_in_range)


                            visible_GOI_tmp=[]
                            for GOI_sep,GOI_pa in zip(sep_real_object_list[GOI_in_range],pa_real_object_list[GOI_in_range]):
                                if GOI_sep>IWA and GOI_sep<OWA:
                                    N_GOI_list[index] = N_GOI_list[index] +1
                                    visible_GOI_tmp.append([star_name,0,compact_date,filter_name,N_cubes,GOI_sep,GOI_pa])
                                    # try:
                                    #     visible_GOI_list[index] = visible_GOI_list[index]+ (star_name,compact_date,filter_name,N_cubes,GOI_sep,GOI_pa)
                                    # except:
                                    #     visible_GOI_list[index] = (star_name,compact_date,filter_name,N_cubes,GOI_sep,GOI_pa)


                            # Loop over all the local maxima stored in the detec csv file
                            for k in range(N_detec):
                                metric_val = detec_table[k,metric_id]
                                x_pos = detec_table[k,x_id]
                                y_pos = detec_table[k,y_id]

                                #remove the detection if it is a real star_name
                                for which_GOI,(x_real_object,y_real_object)  in enumerate(zip(x_real_object_list[GOI_in_range],y_real_object_list[GOI_in_range])):
                                    # print(np.sqrt((x_pos-x_real_object)**2+(y_pos-y_real_object)**2 ))
                                    if (x_pos-x_real_object)**2+(y_pos-y_real_object)**2 < ignore_distance**2:
                                        is_GOI[k] = -1
                                    if (x_pos-x_real_object)**2+(y_pos-y_real_object)**2 < detec_distance**2:
                                        is_GOI[k] = 1
                                        if metric_val > threshold:
                                            N_detected_GOI_list[index] = N_detected_GOI_list[index]+1
                                            visible_GOI_tmp[which_GOI][1] = 1

                            try:
                                visible_GOI_list[index] = visible_GOI_list[index]+ visible_GOI_tmp
                            except:
                                visible_GOI_list[index] = visible_GOI_tmp
                        try:
                            metric_list[index] = metric_list[index]+detec_table[valid_detec[0],metric_id].tolist()
                            pa_list[index] = pa_list[index]+detec_table[valid_detec[0],pa_id].tolist()
                            sep_list[index] = sep_list[index]+detec_table[valid_detec[0],sep_id].tolist()
                            star_name_list[index] = star_name_list[index]+[star_name]*len(detec_table[valid_detec[0],metric_id].tolist())
                            N_cubes_list[index] = N_cubes_list[index]+[N_cubes]*len(detec_table[valid_detec[0],metric_id].tolist())
                            compact_date_list[index] = compact_date_list[index]+[compact_date]*len(detec_table[valid_detec[0],metric_id].tolist())
                            filter_list[index] = filter_list[index]+[filter_name]*len(detec_table[valid_detec[0],metric_id].tolist())
                            is_GOI_list[index] = is_GOI_list[index]+is_GOI[valid_detec[0]].tolist()
                        except:
                            metric_list[index] = detec_table[valid_detec[0],metric_id].tolist()
                            pa_list[index] = detec_table[valid_detec[0],pa_id].tolist()
                            sep_list[index] = detec_table[valid_detec[0],sep_id].tolist()
                            star_name_list[index] = [star_name]*len(detec_table[valid_detec[0],metric_id].tolist())
                            N_cubes_list[index] = [N_cubes]*len(detec_table[valid_detec[0],metric_id].tolist())
                            compact_date_list[index] = [compact_date]*len(detec_table[valid_detec[0],metric_id].tolist())
                            filter_list[index] = [filter_name]*len(detec_table[valid_detec[0],metric_id].tolist())
                            is_GOI_list[index] = is_GOI[valid_detec[0]].tolist()



    for k in range(N_ROC):
        argsort_metric = np.argsort(metric_list[k])[::-1]
        metric_list[k] = np.array(metric_list[k])[argsort_metric].tolist()
        pa_list[k] = np.array(pa_list[k])[argsort_metric].tolist()
        sep_list[k] = np.array(sep_list[k])[argsort_metric].tolist()
        star_name_list[k] = np.array(star_name_list[k])[argsort_metric].tolist()
        N_cubes_list[k] = np.array(N_cubes_list[k])[argsort_metric].tolist()
        compact_date_list[k] = np.array(compact_date_list[k])[argsort_metric].tolist()
        filter_list[k] = np.array(filter_list[k])[argsort_metric].tolist()
        is_GOI_list[k] = np.array(is_GOI_list[k])[argsort_metric].tolist()

    print("N files = {0}".format(N))

    return star_name_list,is_GOI_list,compact_date_list,filter_list,N_cubes_list,metric_list,pa_list,sep_list,N_detected_GOI_list,N_GOI_list,visible_GOI_list