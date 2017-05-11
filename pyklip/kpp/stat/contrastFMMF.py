__author__ = 'jruffio'
import os
import astropy.io.fits as pyfits
from glob import glob
import multiprocessing as mp
import numpy as np
from scipy.signal import convolve2d

import matplotlib.pyplot as plt

from pyklip.kpp.utils.kppSuperClass import KPPSuperClass
from pyklip.kpp.stat.statPerPix_utils import *
from pyklip.kpp.utils.GOI import *
from pyklip.kpp.utils.GPIimage import *

class ContrastFMMF(KPPSuperClass):
    """
    Class for SNR calculation.
    """
    def __init__(self,filename,filename_fakes = None,
                 inputDir = None,
                 outputDir = None,
                 mute=None,
                 N_threads=None,
                 label = None,
                 mask_radius = None,
                 IOWA = None,
                 GOI_list_folder = None,
                 overwrite = False,
                 contrast_filename = None):
        """


        :param filename: Filename of the file on which to calculate the metric. It should be the complete path unless
                        inputDir is defined.
                        It can include wild characters. The file will be selected using the first output of glob.glob().
        :param mute: If True prevent printed log outputs.
        :param N_threads: Number of threads to be used for the metrics and the probability calculations.
                        If None use mp.cpu_count().
                        If -1 do it sequentially.
                        Note that it is not used for this super class.
        :param label: Define the suffix to the output folder when it is not defined. cf outputDir. Default is "default".
        """
        # allocate super class
        super(ContrastFMMF, self).__init__(filename,
                                     inputDir = inputDir,
                                     outputDir = outputDir,
                                     folderName = None,
                                     mute=mute,
                                     N_threads=N_threads,
                                     label=label,
                                     overwrite = overwrite)

        if mask_radius is None:
            self.mask_radius = 3
        else:
            self.mask_radius = mask_radius

        self.IOWA = IOWA
        self.N = 400
        self.Dr = 2
        self.type = "stddev"
        self.suffix = "2Dcontrast"
        self.GOI_list_folder = GOI_list_folder
        self.contrast_filename = contrast_filename
        self.filename_fakes = filename_fakes


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
        init_out = super(ContrastFMMF, self).initialize(inputDir = inputDir,
                                         outputDir = outputDir,
                                         folderName = folderName,
                                         label=label)

        if self.contrast_filename is not None:
            # Check file existence and define filename_path
            if self.inputDir is None or os.path.isabs(self.contrast_filename):
                try:
                    if len(glob(self.contrast_filename)) == self.N_matching_files:
                        self.contrast_filename_path = os.path.abspath(glob(self.contrast_filename)[self.id_matching_file-1])
                    else:
                        self.contrast_filename_path = os.path.abspath(glob(self.contrast_filename)[0])
                except:
                    raise Exception("File "+self.contrast_filename+"doesn't exist.")
            else:
                try:
                    if len(glob(self.inputDir+os.path.sep+self.contrast_filename)) == self.N_matching_files:
                        self.contrast_filename_path = os.path.abspath(glob(self.inputDir+os.path.sep+self.contrast_filename)[self.id_matching_file-1])
                    else:
                        self.contrast_filename_path = os.path.abspath(glob(self.inputDir+os.path.sep+self.contrast_filename)[0])
                except:
                    raise Exception("File "+self.inputDir+os.path.sep+self.contrast_filename+" doesn't exist.")

        if self.filename_fakes is not None:
            # Check file existence and define filename_path
            if self.inputDir is None or os.path.isabs(self.filename_fakes):
                try:
                    self.filename_fakes_path = os.path.abspath(glob(self.filename_fakes)[self.id_matching_file])
                except:
                    raise Exception("File "+self.filename_fakes+"doesn't exist.")
            else:
                try:
                    self.filename_fakes_path = os.path.abspath(glob(self.inputDir+os.path.sep+self.filename_fakes)[self.id_matching_file])
                except:
                    raise Exception("File "+self.inputDir+os.path.sep+self.filename_fakes+" doesn't exist.")

            # Open the fits file on which the metric will be applied
            hdulist = pyfits.open(self.filename_fakes_path)
            if not self.mute:
                print("Opened: "+self.filename_fakes_path)

            # grab the data and headers
            try:
                self.image_fakes = hdulist[1].data
                self.exthdr_fakes = hdulist[1].header
                self.prihdr_fakes = hdulist[0].header
            except:
                raise Exception("Couldn't read "+self.filename_fakes_path+". Is it a .fits file with GPI extensions convention?")

        # Get center of the image (star position)
        try:
            # Retrieve the center of the image from the fits headers.
            self.center = [self.exthdr['PSFCENTX'], self.exthdr['PSFCENTY']]
        except:
            # If the keywords could not be found the center is defined as the middle of the image
            if not self.mute:
                print("Couldn't find PSFCENTX and PSFCENTY keywords.")
            self.center = [(self.nx-1)/2,(self.ny-1)/2]


        try:
            self.folderName = self.exthdr["METFOLDN"]+os.path.sep
        except:
            pass

        file_ext_ind = os.path.basename(self.filename_path)[::-1].find(".")
        self.prefix = os.path.basename(self.filename_path)[:-(file_ext_ind+1)]

        return init_out

    def check_existence(self):
        """

        :return: False
        """

        file_exist = (len(glob(self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+self.suffix+'.fits')) >= 1)

        if file_exist and not self.mute:
            print("Output already exist: "+self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+self.suffix+'.fits')

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


        # Evaluate the throughput of the metric
        if self.filename_fakes is not None:
            row_real_object_list,col_real_object_list = get_pos_known_objects(self.prihdr_fakes,self.exthdr_fakes,fakes_only=True)
            sep_list,pa_real_object_list = get_pos_known_objects(self.prihdr_fakes,self.exthdr_fakes,pa_sep=True,fakes_only=True)
            print(sep_list)
            real_contrast_list = []
            for fake_id in range(100):
                try:
                    real_contrast_list.append(self.exthdr_fakes["FKCONT{0:02d}".format(fake_id)])
                except:
                    continue
            print(real_contrast_list)
            # flux_list = []
            # metric_list = []
            self.metric_list = [self.image_fakes[np.round(row_real_object),np.round(col_real_object)] \
                                         for row_real_object,col_real_object in zip(row_real_object_list,col_real_object_list)]
            where_nans = np.where((~np.isnan(self.metric_list))*(sep_list>0.5))
            self.metric_list = np.array(self.metric_list)[where_nans]
            sep_list =  np.array(sep_list)[where_nans]
            real_contrast_list =  np.array(real_contrast_list)[where_nans]

            self.throughput = np.mean(np.array(real_contrast_list)/np.array(self.metric_list))
            print("THROUGHPUT = {0}".format(self.throughput))
            self.throughput = 2.1

            print(self.metric_list)
            print(sep_list)
            print(real_contrast_list)
            import matplotlib.pyplot as plt
            plt.plot(sep_list,np.array(self.metric_list)/np.array(real_contrast_list))
            plt.show()
            print(np.array(self.metric_list)/3.e-6)
            exit()

        else:
            self.throughput = 1
            if not self.mute:
                print("No file with fake planets has been provided to estimate the throughput. Assuming 1.")


        # If GOI_list_folder is not None. Mask the known objects from the image that will be used for calculating the
        # PDF. This masked image is given separately to the probability calculation function.
        if self.GOI_list_folder is not None:
            self.image_without_planet = mask_known_objects(self.image,self.prihdr,self.exthdr,self.GOI_list_folder, mask_radius = self.mask_radius)
        else:
            self.image_without_planet = self.image

        self.flux_1Dstddev,self.flux_stddev_rSamp = get_image_stddev(self.image_without_planet,
                                                                     self.IOWA,
                                                                     N = None,
                                                                     centroid = self.center,
                                                                     r_step = self.Dr,
                                                                     Dr=self.Dr)
        self.flux_stddev_rSamp = np.array([r_tuple[0] for r_tuple in self.flux_stddev_rSamp])
        self.flux_1Dstddev = np.array(self.flux_1Dstddev)
        # self.flux_1Dstddev_map = get_image_stat_map(self.image,
        #                                             self.image_without_planet,
        #                                             IOWA = self.IOWA,
        #                                             N = None,
        #                                             centroid = self.center,
        #                                             r_step = self.Dr/2,
        #                                             Dr = self.Dr,
        #                                             type = "stddev",
        #                                             image_wide = None)
        #
        #
        # self.fluxMap_stddev = get_image_stat_map_perPixMasking(self.image,
        #                                                  self.image_without_planet,
        #                                                  mask_radius = self.mask_radius,
        #                                                  IOWA = self.IOWA,
        #                                                  N = self.N,
        #                                                  centroid = self.center,
        #                                                  mute = self.mute,
        #                                                  N_threads = self.N_threads,
        #                                                  Dr= self.Dr,
        #                                                  Dth = None,
        #                                                  type = self.type)


        legend_str_list = []
        plt.figure(1,figsize=(8,6))
        # plt.subplot(1,2,1)
        if self.contrast_filename is not None:
            with open(self.contrast_filename_path, 'rt') as cvs_contrast:
                cvs_contrast_reader = csv.reader(filter(lambda row: row[0]!="#",cvs_contrast),delimiter=' ')
                list_contrast = list(cvs_contrast_reader)
                contrast_str_arr = np.array(list_contrast, dtype='string')
                col_names = contrast_str_arr[0]
                contrast_arr = contrast_str_arr[1::].astype(np.float)
                self.sep_samples = contrast_arr[:,0]
                self.Ttype_contrast = np.squeeze(contrast_arr[:,np.where("T-Type"==col_names)])
                self.Ltype_contrast = np.squeeze(contrast_arr[:,np.where("L-Type"==col_names)])


                plt.plot(self.sep_samples,self.Ttype_contrast,"--", color='b', linewidth=3.0)
                legend_str_list.append("T-type pyklip")
                plt.plot(self.sep_samples,self.Ltype_contrast,"--", color='r', linewidth=3.0)
                legend_str_list.append("L-type pyklip")

        plt.plot(self.flux_stddev_rSamp*0.01413,5*self.flux_1Dstddev*self.throughput, color='r', linewidth=3.0)
        legend_str_list.append("{0} FMpF".format(self.folderName))
        plt.xlabel("Separation (arcsec)", fontsize=20)
        plt.ylabel("Contrast (log10)", fontsize=20)
        plt.legend(legend_str_list)
        ax= plt.gca()
        ax.set_yscale('log')
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        # ax.xaxis.set_ticks_position('bottom')
        # ax.yaxis.set_ticks_position('left')

        # plt.subplot(1,2,2)
        # plt.imshow(5*(self.fluxMap_stddev-self.flux_1Dstddev_map))
        # plt.colorbar()
        # ax = plt.gca()
        # # Remove box and axes ticks
        # ax.set_axis_off()
        # # rect = fig.patch
        # # rect.set_facecolor('white')

        # return self.fluxMap_stddev
        return None

    def save(self):
        """

        :return: None
        """

        if not os.path.exists(self.outputDir+os.path.sep+self.folderName):
            os.makedirs(self.outputDir+os.path.sep+self.folderName)

        self.suffix = "1Dcontrast"
        if not self.mute:
            print("Saving: "+self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+self.suffix+'.csv')
        with open(self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+self.suffix+'.csv', 'w+') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=' ')
            csvwriter.writerows([["Seps","T-Type"]])
            csvwriter.writerows(zip(self.flux_stddev_rSamp*0.01413,5*self.flux_1Dstddev*self.throughput))

        if not self.mute:
            print("Saving: "+self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+self.suffix+'.png')
        plt.savefig(self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+self.suffix+".png", bbox_inches='tight')

        # if hasattr(self,"prihdr") and hasattr(self,"exthdr"):
        #     # Save the parameters as fits keywords
        #     # STA##### stands for STAtistic
        #     self.exthdr["STA_TYPE"] = self.type
        #
        #     self.exthdr["STAFILEN"] = self.filename_path
        #     self.exthdr["STAINDIR"] = self.inputDir
        #     self.exthdr["STAOUTDI"] = self.outputDir
        #     self.exthdr["STAFOLDN"] = self.folderName
        #
        #     self.exthdr["STAMASKR"] = self.mask_radius
        #     self.exthdr["STA_IOWA"] = str(self.IOWA)
        #     self.exthdr["STA_N"] = self.N
        #     self.exthdr["STA_DR"] = self.Dr
        #     self.exthdr["STA_TYPE"] = self.type
        #     self.exthdr["STAGOILF"] = self.GOI_list_folder
        #
        #     # # This parameters are not always defined
        #     # if hasattr(self,"spectrum_name"):
        #     #     self.exthdr["STASPECN"] = self.spectrum_name
        #
        #     self.suffix = "2Dcontrast"
        #     if not self.mute:
        #         print("Saving: "+self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+self.suffix+'.fits')
        #     hdulist = pyfits.HDUList()
        #     hdulist.append(pyfits.PrimaryHDU(header=self.prihdr))
        #     hdulist.append(pyfits.ImageHDU(header=self.exthdr, data=self.fluxMap_stddev, name=self.suffix))
        #     hdulist.writeto(self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+self.suffix+'.fits', clobber=True)
        # else:
        #     hdulist = pyfits.HDUList()
        #     hdulist.append(pyfits.ImageHDU(data=self.fluxMap_stddev, name=self.suffix))
        #     hdulist.append(pyfits.ImageHDU(name=self.suffix))
        #
        #     hdulist[1].header["STA_TYPE"] = self.type
        #
        #     hdulist[1].header["STAFILEN"] = self.filename_path
        #     hdulist[1].header["STAINDIR"] = self.inputDir
        #     hdulist[1].header["STAOUTDI"] = self.outputDir
        #     hdulist[1].header["STAFOLDN"] = self.folderName
        #
        #     hdulist[1].header["STAMASKR"] = self.mask_radius
        #     hdulist[1].header["STA_IOWA"] = self.IOWA
        #     hdulist[1].header["STA_N"] = self.N
        #     hdulist[1].header["STA_DR"] = self.Dr
        #     hdulist[1].header["STA_TYPE"] = self.type
        #     hdulist[1].header["STAGOILF"] = self.GOI_list_folder
        #
        #     self.suffix = "2Dcontrast"
        #     if not self.mute:
        #         print("Saving: "+self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+self.suffix+'.fits')
        #     hdulist.writeto(self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+self.suffix+'.fits', clobber=True)

        # plt.close(1)
        plt.show()
        return None

    def load(self):
        """

        :return: None
        """

        return None



def gather_contrasts(base_dir,filename_filter_list,mute = False,epoch_suffix=None, cont_name_list = None,band=None,stars2ignore=None):
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

    N_cont = len(filename_filter_list)
    sep_samp_list = [[]]*N_cont
    cont_list = [[]]*N_cont
    star_name_list = [[]]*N_cont

    if epoch_suffix is None:
        epoch_suffix = ""

    if cont_name_list is None:
        cont_name_list = ["T-Type",]*N_cont

    if band is None:
        band = "H"

    if stars2ignore is None:
        stars2ignore=[]

    dirs_to_reduce = os.listdir(base_dir)
    # dirs_to_reduce = ["HD_90885_B"]
    # dirs_to_reduce = ["c_Eri"]
    N=0
    for object in dirs_to_reduce:
        if not object.startswith('.') and object not in stars2ignore:
            print(object)

            epochDir_glob = glob(base_dir+object+os.path.sep+"autoreduced_kpop"+os.path.sep+"*_{0}_Spec".format(band)+epoch_suffix+os.path.sep)

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
                        # print(inputDir+os.path.sep+filename_filter)
                        # print(glob(inputDir+os.path.sep+filename_filter)[0])
                        file_list.append(glob(inputDir+os.path.sep+filename_filter)[0])
                        # if not mute:
                        #     print("ROC: {0} in {1}. Adding.".format(filename_filter,inputDir))
                    except:
                        pass
                        # if not mute:
                        #     print("ROC: {0} unvailable in {1}. Skipping".format(filename_filter,inputDir))

                if len(file_list) == N_cont:
                    # print(file_list)
                    N=N+1
                    for index,(filename,cont_name) in enumerate(zip(file_list,cont_name_list)):
                        with open(filename, 'rb') as csvfile:
                            # reader = csv.reader(csvfile, delimiter=' ')
                            # csv_as_list = list(reader)
                            # detec_table_labels = csv_as_list[0]
                            # detec_table = np.array(csv_as_list[1::], dtype='string').astype(np.float)

                            cvs_contrast_reader = csv.reader(filter(lambda row: row[0]!="#",csvfile),delimiter=' ')
                            list_contrast = list(cvs_contrast_reader)
                            # print(list_contrast)
                            contrast_str_arr = np.array(list_contrast, dtype='string')
                            # print(contrast_str_arr)
                            col_names = contrast_str_arr[0]
                            contrast_arr = contrast_str_arr[1::].astype(np.float)
                            sep_samples = contrast_arr[:,0]


                            methane_idx = np.where(cont_name==col_names)[0]

                            methane_contrast = np.squeeze(contrast_arr[:,methane_idx])

                        if N == 1:
                            cont_list[index] = [methane_contrast]
                            sep_samp_list[index] = [sep_samples]
                            star_name_list[index] = [object]
                        else:
                            cont_list[index].append(methane_contrast)
                            sep_samp_list[index].append(sep_samples)
                            star_name_list[index].append(object)


                        # try:
                        #     cont_list[index] = cont_list[index]+methane_contrast
                        # except:
                        #     sep_samp_list[index] = sep_samples
                        #     cont_list[index] = methane_contrast

    print("N files = {0}".format(N))


    final_sep_samp_list = []
    for k in range(N_cont):
        sep_samp_big_vector = np.concatenate(sep_samp_list[k])
        final_sep_samp_list.append(np.unique(sep_samp_big_vector))


    N_curves = len(cont_list[0])
    final_cont_list = []
    for k in range(N_cont):
        final_sep_samp = final_sep_samp_list[k]
        curr_cont_list = np.zeros((N_curves,np.size(final_sep_samp)))+np.nan
        for l,(sep_samp,cont,star_name) in enumerate(zip(sep_samp_list[k],cont_list[k],star_name_list[k])):
            print(star_name)
            interp_f = interp1d(sep_samp,cont,bounds_error = False,fill_value=np.nan)
            curr_cont_list[l,:] = interp_f(final_sep_samp)
            # print(l,ind,(ind+np.size(cont)))
            # print(cont)
            # print(curr_cont_list[l,:])
        final_cont_list.append(curr_cont_list)

    # return sep_samp_list,np.array(cont_list)/N
    return final_sep_samp_list,final_cont_list,star_name_list
