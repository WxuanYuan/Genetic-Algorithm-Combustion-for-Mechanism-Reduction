import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class SimData:

    def __init__(self, RRCNum, GroNum, PointNumMax):
        self.RanN = np.zeros(RRCNum)  # 随机数
        self.tarM = np.zeros((GroNum, PointNumMax))  # 仿真target
        self.Etar = np.zeros((GroNum, PointNumMax))  # 目标仿真误差
        self.Egro = np.zeros((GroNum))  # 每组平均仿真误差


class IDT_Label:
    """This class reads and stores the IDT experimental data. """

    def __init__(self, root_path, IDT_configuration_path="CH4_IDT_configuration.dat", experimental_IDT_path="CH4_experimental_IDT.csv"):
        """Reads and stores the IDT experimental data.

        Parameters
        ----------
        root_path: String
            Root path of the program.
        IDT_configuration_path: String
            Name of the IDT configuration file.
        experimental_IDT_path: String
            Name of the IDT experimental data file.
        """
        self.description = "IDP Error"

        # read parameters
        try:
            with open(root_path + "\source_files" + f"\{IDT_configuration_path}", 'r') as input_data:
                self.RRCNum = int((input_data.readline()).split(' ', 1)[0])   # RRC个数
                self.FuelNum = int((input_data.readline()).split(' ', 1)[0])  # 组分
                self.GroNum = int((input_data.readline()).split(' ', 1)[0])  # 试验总组数
                self.PointNumMax = int((input_data.readline()).split(' ', 1)[0])  # 每组最大试验点数
                self.MechName = str((input_data.readline()).split(' ', 1)[0])  # 机理文件名
                self.PeakName = str((input_data.readline()).split(' ', 1)[0])  # IDT判断峰值组分名
            print(f"Experimental IDT loaded from path [source_files\\{IDT_configuration_path}].")
        except FileNotFoundError:
            print(f"No experimental IDT found under path [source_files\\{IDT_configuration_path}].")
            raise FileNotFoundError

        # read data
        self.T5 = np.zeros((self.GroNum, self.PointNumMax))
        self.p5 = np.zeros((self.GroNum, self.PointNumMax))
        self.FuelName = np.empty(self.FuelNum, dtype=object)
        self.FuelMF = np.zeros((self.GroNum, self.PointNumMax, self.FuelNum))
        self.Phi = np.zeros((self.GroNum, self.PointNumMax))
        self.ExpIDT = np.zeros((self.GroNum, self.PointNumMax))
        self.IDTrun = np.zeros((self.GroNum, self.PointNumMax))
        self.ExpUn = np.zeros((self.GroNum, self.PointNumMax))
        self.SimIDT = np.zeros((self.GroNum, self.PointNumMax, 5))
        self.IDTmethod = np.empty((self.GroNum, self.PointNumMax), dtype=object)
        data = pd.read_csv(root_path + "\source_files" + f"\{experimental_IDT_path}")
        Name = data.columns.values
        for I in range(self.FuelNum):
            self.FuelName[I] = str(Name[I + 4])

        for I in range(data.shape[0]):
            II = int(data['GroNum'][I]) - 1
            JJ = int(data['PointNum'][I]) - 1
            # read pressure, temperature and species concentration configurations for each point in each group
            self.T5[II, JJ] = float(data['T5'][I])
            self.p5[II, JJ] = float(data['p5'][I])
            for J in range(self.FuelNum):
                self.FuelMF[II, JJ, J] = float(data[str(self.FuelName[J])][I])

            self.Phi[II, JJ] = float(data['Phi'][I])
            self.ExpIDT[II, JJ] = float(data['ExpIDT'][I])
            self.IDTrun[II, JJ] = float(data['IDTrun'][I])
            self.ExpUn[II, JJ] = float(data['ExpUn'][I])
            self.IDTmethod[II, JJ] = str(data['Condition'][I])
        self.IDTmethod = self.IDTmethod.tolist()

        self.results = SimData(self.RRCNum, self.GroNum, self.PointNumMax)


class PFR_Label:
    """This class Reads and stores the PFR experimental data. """

    def __init__(self, root_path, PFR_configuration_path="CH4_PFR_configuration.dat", experimental_PFR_path="CH4_experimental_PFR.csv"):
        """This class Reads and stores the PFR experimental data.

        Parameters
        ----------
        root_path: String
            Root path of the program.
        PFR_configuration_path: String
            Name of the PFR configuration file.
        experimental_PFR_path: String
            Name of the PFR experimental data file.
        """
        self.description = "PFR Error"

        # read parameters
        try:
            with open(root_path + "\source_files" + f"\{PFR_configuration_path}", 'r') as input_data:
                self.RRCNum = int((input_data.readline()).split(' ', 1)[0])  # RRC个数
                self.FuelNum = int((input_data.readline()).split(' ', 1)[0])  # 组分
                self.GroNum = int((input_data.readline()).split(' ', 1)[0])  # 试验总组数
                self.PointNumMax = int((input_data.readline()).split(' ', 1)[0])  # 每组最大试验点数
                self.MechName = str((input_data.readline()).split(' ', 1)[0])  # 机理文件名
                self.timesteps = int((input_data.readline()).split(' ', 1)[0])
                self.PeakName_1 = str((input_data.readline()).split(' ', 1)[0])  # IDT判断峰值组分名1
                self.PeakName_2 = str((input_data.readline()).split(' ', 1)[0])  # IDT判断峰值组分名2
            print(f"Experimental PFR loaded from path [source_files\\{PFR_configuration_path}].")
        except FileNotFoundError:
            print(f"No experimental PFR found under path [source_files\\{PFR_configuration_path}].")
            raise FileNotFoundError

        # initialize and read data
        self.T5 = np.zeros((self.GroNum, self.PointNumMax))
        self.p5 = np.zeros((self.GroNum, self.PointNumMax))
        self.FuelName = np.empty(self.FuelNum, dtype=object)
        self.FuelMF = np.zeros((self.GroNum, self.PointNumMax, self.FuelNum))
        # self.Phi = np.zeros((self.GroNum, self.PointNumMax))
        self.Simdata = np.zeros((self.GroNum, self.PointNumMax))
        self.Runtime = np.zeros((self.GroNum, self.PointNumMax))
        self.Exptime = np.zeros((self.GroNum, self.PointNumMax))
        self.ExpUn = np.zeros((self.GroNum, self.PointNumMax))
        self.Expdata = np.zeros((self.GroNum, self.PointNumMax))
        self.SpecieName1 = np.empty(self.GroNum, dtype=object)
        # self.SpecieName1 = np.zeros((self.GroNum, self.PointNumMax))
        self.t = np.zeros((self.GroNum, self.PointNumMax))

        data = pd.read_csv(root_path + "\source_files" + f"\{experimental_PFR_path}")
        Name = data.columns.values
        for I in range(self.FuelNum):
            self.FuelName[I] = str(Name[I + 4])

        for I in range(data.shape[0]):
            II = int(data['GroNum'][I]) - 1
            JJ = int(data['PointNum'][I]) - 1
            self.T5[II, JJ] = float(data['T5'][I])
            self.p5[II, JJ] = float(data['p5'][I])

            for J in range(self.FuelNum):
                self.FuelMF[II, JJ, J] = float(data[str(self.FuelName[J])][I])

            self.Simdata[II, JJ] = float(data['Simdata'][I])
            self.Runtime[II, JJ] = float(data['Runtime'][I])
            self.Exptime[II, JJ] = float(data['Exptime'][I])
            self.ExpUn[II, JJ] = float(data['ExpUn'][I])
            self.Expdata[II, JJ] = float(data['Expdata'][I])
            self.SpecieName1[II] = str(data['specie'][I])
        self.SpecieName1 = self.SpecieName1.tolist()

        self.results = SimData(self.RRCNum, self.GroNum, self.PointNumMax)

