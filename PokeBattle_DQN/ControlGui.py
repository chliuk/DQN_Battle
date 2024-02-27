from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from cv2 import IMREAD_LOAD_GDAL
from UI import Ui_MainWindow
from Q1 import find_corner,find_intrinsic,find_extrinsic,find_distortion,show_result
from Q2 import Q2_1,Q2_2
from Q3 import stereo_map
import sys
class Window(QMainWindow, Ui_MainWindow):
    global folder_path
    global Image_L
    global Image_R
    def __init__(self): 
        super(Window, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        self.ui.pushButton_6.clicked.connect(lambda: find_extrinsic(int(self.ui.comboBox.currentText())))
        #Button connect open_folder
        self.ui.pushButton.clicked.connect(self.open_folder)
        #Button connect open_LImage
        self.ui.pushButton_2.clicked.connect(self.open_LImage)
        #Button connect open_RImage
        self.ui.pushButton_3.clicked.connect(self.open_RImage)

        #Q1
        #Button 1.1 ～ 1.5
        self.ui.pushButton_4.clicked.connect(find_corner)
        self.ui.pushButton_5.clicked.connect(find_intrinsic)
        self.ui.pushButton_6.clicked.connect(lambda: find_extrinsic(int(self.ui.comboBox.currentText())))
        self.ui.pushButton_7.clicked.connect(find_distortion)
        self.ui.pushButton_8.clicked.connect(show_result)

        #Q2
        #Button 2.1 ～ 2.2
        self.ui.pushButton_9.clicked.connect(lambda: Q2_1(str(self.ui.lineEdit.text())))
        self.ui.pushButton_10.clicked.connect(lambda: Q2_2(str(self.ui.lineEdit.text())))

        #Q3
        self.ui.pushButton_11.clicked.connect(lambda: stereo_map(str(Image_L), str(Image_R)))

    def open_folder(self):
        global folder_path
        folder_path = QFileDialog.getExistingDirectory(self, "Open folder", "./")
    
    def open_LImage(self):
        global Image_L
        filename = QFileDialog.getOpenFileName(self, "Open file", "./")
        Image_L = filename

    def open_RImage(self):
        global Image_R
        filename = QFileDialog.getOpenFileName(self, "Open file", "./")
        Image_R = filename

    

if __name__ == '__main__':
    app = QApplication(sys.argv)
    Win = Window() 
    Win.show()  
    sys.exit(app.exec_())
