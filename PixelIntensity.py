import cv2
import numpy as np
import options as options
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog, QFormLayout, QPushButton, QLineEdit, QComboBox, QDialog, QLabel
from cv2.cv2 import QT_PUSH_BUTTON
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt

class Ui_MainWindow(QtWidgets.QWidget): #object
    image1path=''
    image2path=''
    transformedimage=[]

#GUI for Image Transformation
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(-1, -1, 801, 561))
        self.widget.setObjectName("widget")
        self.formLayoutWidget = QtWidgets.QWidget(self.widget)
        self.formLayoutWidget.setGeometry(QtCore.QRect(200, 0, 601, 561))
        self.formLayoutWidget.setObjectName("formLayoutWidget")
        self.formLayout = QtWidgets.QFormLayout(self.formLayoutWidget)
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.formLayout.setObjectName("formLayout")
        self.DefaultImage = QtWidgets.QLabel(self.formLayoutWidget)
        self.DefaultImage.setFixedSize(512,512)
        self.DefaultImage.setObjectName("DefaultImage")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.DefaultImage)
        self.verticalLayoutWidget = QtWidgets.QWidget(self.widget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(0, 10, 199, 551))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.LoadImage = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.LoadImage.setAutoDefault(False)
        self.LoadImage.setObjectName("LoadImage")
        self.verticalLayout.addWidget(self.LoadImage)
        self.ImageNegative = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.ImageNegative.setObjectName("ImageNegative")
        self.verticalLayout.addWidget(self.ImageNegative)
        self.BitPlane = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.BitPlane.setObjectName("BitPlane")
        self.verticalLayout.addWidget(self.BitPlane)
        self.LogTransform = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.LogTransform.setObjectName("LogTransform")
        self.verticalLayout.addWidget(self.LogTransform)
        self.GammaTransform = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.GammaTransform.setObjectName("GammaTransform")
        self.verticalLayout.addWidget(self.GammaTransform)
        self.Linear = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.Linear.setObjectName("Linear")
        self.verticalLayout.addWidget(self.Linear)
        self.PiecewiseLinear = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.PiecewiseLinear.setObjectName("PiecewiseLinear")
        self.verticalLayout.addWidget(self.PiecewiseLinear)
        self.Arithmetic = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.Arithmetic.setObjectName("Arithmetic")
        self.verticalLayout.addWidget(self.Arithmetic)
        self.SetOperation = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.SetOperation.setObjectName("SetOperation")
        self.verticalLayout.addWidget(self.SetOperation)
        self.Thresholding = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.Thresholding.setObjectName("Thresholding")
        self.verticalLayout.addWidget(self.Thresholding)
        self.Binarization = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.Binarization.setObjectName("Binarization")
        self.verticalLayout.addWidget(self.Binarization)
        self.Logical = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.Logical.setObjectName("Logical")
        self.verticalLayout.addWidget(self.Logical)
        self.SaveTransform = QtWidgets.QPushButton(self.centralwidget)
        self.SaveTransform.setGeometry(QtCore.QRect(510, 550, 201, 32))
        self.SaveTransform.setObjectName("SaveTransform")
        self.RefreshImage = QtWidgets.QPushButton(self.centralwidget)
        self.RefreshImage.setGeometry(QtCore.QRect(260, 550, 201, 32))
        self.RefreshImage.setObjectName("RefreshImage")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

#Connect functions to GUI
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Assignment3 "))
        self.DefaultImage.setText(_translate("MainWindow", "INSTRUCTIONS\n \
        Load New Image: Select an image to perform transformations\n \
        Select a transformation from the menu\n \
        Refresh Image: Resets Transformations to original image\n \
        Save Transformation: Enables User to save the transformed image"))
        self.LoadImage.setText(_translate("MainWindow", "Load New Image"))
        self.ImageNegative.setText(_translate("MainWindow", "Image negative"))
        self.BitPlane.setText(_translate("MainWindow", "Bit-plane images"))
        self.LogTransform.setText(_translate("MainWindow", "Log transformation"))
        self.Thresholding.setText(_translate("MainWindow", "Thresholding"))
        self.GammaTransform.setText(_translate("MainWindow", "Gamma transformation"))
        self.Linear.setText(_translate("MainWindow", "Linear"))
        self.PiecewiseLinear.setText(_translate("MainWindow", "Piecewise-linear"))
        self.Arithmetic.setText(_translate("MainWindow", "Arithmetic operations"))
        self.SetOperation.setText(_translate("MainWindow", "Set operations"))
        self.Binarization.setText(_translate("MainWindow", "Binarization"))
        self.Logical.setText(_translate("MainWindow", "Logical operations"))
        self.SaveTransform.setText(_translate("MainWindow", "Save Transformed Image"))
        self.RefreshImage.setText(_translate("MainWindow", "Refresh Image"))

        self.SaveTransform.setCheckable(True)
        self.SaveTransform.clicked.connect(lambda:self.saveimage(transformedimage))

        self.RefreshImage.setCheckable(True)
        self.RefreshImage.clicked.connect(lambda:self.refreshimage())

        self.LoadImage.setCheckable(True)
        self.LoadImage.clicked.connect(lambda:self.openfilebrowser())

        self.BitPlane.setCheckable(True)
        self.BitPlane.clicked.connect(lambda:self.bitplane())

        self.LogTransform.setCheckable(True)
        self.LogTransform.clicked.connect(lambda:self.log())

        self.Arithmetic.setCheckable(True)
        self.Arithmetic.clicked.connect(lambda:self.arithmetic())

        self.GammaTransform.setCheckable(True)
        self.GammaTransform.clicked.connect(lambda: self.gamma())


        self.ImageNegative.setCheckable(True)
        self.ImageNegative.clicked.connect(lambda: self.negative())

        self.Logical.setCheckable(True)
        self.Logical.clicked.connect(lambda: self.logical())

        self.Binarization.setCheckable(True)
        self.Binarization.clicked.connect(lambda:self.binarization())

        self.Thresholding.setCheckable(True)
        self.Thresholding.clicked.connect(lambda:self.threshold())

        self.Linear.setCheckable(True)
        self.Linear.clicked.connect(lambda:self.linear())

        self.PiecewiseLinear.setCheckable(True)
        self.PiecewiseLinear.clicked.connect(lambda:self.piecewiselinear())

        self.SetOperation.setCheckable(True)
        self.SetOperation.clicked.connect(lambda:self.setoperation())

#Opens the file browser to select an image
    def openfilebrowser(self):
        options=QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName = QFileDialog.getOpenFileName(None,"QFileDialog.getOpenFileName()", "","Image files (*.jpg *.gif *.png *.jpeg)", options=options)
        global image1path
        image1path=fileName[0]
        self.img=cv2.imread(image1path)
        self.showimage(self.img)

#Displays the image in the main GUI
    def showimage(self,img):

        global transformedimage
        transformedimage=img
        self.dispimage=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        self.img_resize = cv2.resize(self.dispimage, (512, 512))
        self.img_resize = QtGui.QImage(self.img_resize.data, self.img_resize.shape[1], self.img_resize.shape[0],
                                       self.img_resize.strides[0],
                                       QtGui.QImage.Format_RGB888)
        self.DefaultImage.setPixmap(QPixmap.fromImage(self.img_resize))

#Refreshs the image in the GUI to the original image
    def refreshimage(self):
        img=cv2.imread(image1path)
        self.showimage(img)

#Artithmetic Operation Dialog box
    def arithmetic(self):
    #image arithmetic
        arithmetic_operations = ["Addition","Subtraction","Multiplication","Division"]
        dialog=QDialog()
        dialog.setModal(True)
        combobox=QComboBox(dialog)
        combobox.setGeometry(QtCore.QRect(100, 40, 201, 22))
        combobox.addItems(arithmetic_operations)
        button_selectimage2=QPushButton("Select Second Image",dialog)
        button_selectimage2.clicked.connect(self.selectimage2)
        button_transform = QPushButton("Apply Transformation", dialog)
        button_transform.clicked.connect(lambda:self.arithmetic_apply(combobox.currentText()))
        button_transform.move(200, 70)
        dialog.setFixedSize(400,100)
        dialog.exec_()

#Gamma operation Dialog box
    def gamma(self):
        dialog = QDialog()
        dialog.setModal(True)
        text_field=QLineEdit(dialog)
        text_field.setMaxLength(3)
        text_field.setPlaceholderText("ENTER C VALUE")
        text_field.move(200,10)
        text_field1=QLineEdit(dialog)
        text_field1.setMaxLength(3)
        text_field1.move(10,10)
        text_field1.setPlaceholderText("ENTER GAMMA VALUE")
        text_field1.setFixedWidth(180)
        button_transform = QPushButton("Apply Transformation", dialog)
        button_transform.move(50, 70)
        button_transform.clicked.connect(lambda:self.gammatransform(text_field1.text(),text_field.text()))
        dialog.setFixedSize(400, 100)
        dialog.exec_()

#Open file browser to select image 2
    def selectimage2(self):
        options=QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName = QFileDialog.getOpenFileName(None,"QFileDialog.getOpenFileName()", "","Image files (*.jpg *.gif *.png *.jpeg)", options=options)
        global image2path
        image2path=fileName[0]
        image2=cv2.imread(image2path)
        window=cv2.namedWindow('ImageTwo',cv2.WINDOW_GUI_EXPANDED)
        cv2.imshow(window,image2)

#Provide user the option to save the image
    def saveimage(self,transformed_image):
        global  transformedimage
        transformedimage=transformed_image
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        img = np.array(transformedimage)
        filePath, _ = QFileDialog.getSaveFileName(self, "Save Image", "",
                                                      "PNG(*.png);;JPEG(*.jpg *.jpeg);;All Files(*.*) ")
        if(filePath!=''):
            cv2.imwrite(filePath, img)

#Apply arithmetic operations to the two image
    def arithmetic_apply(self, combobox):
        cv2.destroyAllWindows()
        global image2path
        global image1path
        img = cv2.imread(image1path)
        img1 = cv2.imread(image2path)
        img_resize = cv2.resize(img, (512, 512))
        img1_resize = cv2.resize(img1, (512, 512))
        if(combobox=='Addition'):
            add = cv2.add(img, img1)
            self.showimage(add)


        if(combobox=='Subtraction'):
            subtract = cv2.subtract(img, img1)
            self.showimage(subtract)


        if(combobox=='Multiplication'):
            multiply = cv2.multiply(img, img1)
            self.showimage(multiply)

        if(combobox=='Division'):
            divide = cv2.divide(img, img1)
            self.showimage(divide)

#Apply negative transformation on image
    def negative(self):
        p = cv2.imread(image1path)
        negative_img = 1 - p
        self.showimage(negative_img)

#Logical operations Dialog Box
    def logical(self):
        Logical_operations = ["AND","OR","XOR","NOT"]
        dialog=QDialog()
        dialog.setModal(True)
        combobox=QComboBox(dialog)
        combobox.setGeometry(QtCore.QRect(100, 40, 201, 22))
        combobox.addItems(Logical_operations)
        button_selectimage2=QPushButton("Select Second Image",dialog)
        button_selectimage2.clicked.connect(self.selectimage2)
        button_transform = QPushButton("Apply Transformation", dialog)
        button_transform.clicked.connect(lambda:self.logical_apply(combobox.currentText()))
        button_transform.move(200, 70)
        dialog.setFixedSize(400,100)
        dialog.exec_()

#Apply Logical operations on the images
    def logical_apply(self,combobox):
        cv2.destroyAllWindows()
        global image2path
        global image1path
        img = cv2.imread(image1path)
        img1 = cv2.imread(image2path)
        img_resize = cv2.resize(img, (512, 512))
        img1_resize = cv2.resize(img1, (512, 512))
        if (combobox == 'AND'):
            logic_and = cv2.bitwise_and(img1, img, mask = None)
            self.showimage(logic_and)

        if (combobox == 'OR'):
            logic_or = cv2.bitwise_or(img1, img, mask = None)
            self.showimage(logic_or)

        if (combobox == 'XOR'):
            logic_xor = cv2.bitwise_xor(img1, img)
            self.showimage(logic_xor)


        if (combobox == 'NOT'):
            logic_not1=cv2.bitwise_not(img)
            logic_not2=cv2.bitwise_not(img1)
            numpy_horizontal=np.hstack((logic_not1,logic_not2))
            self.showimage(numpy_horizontal)

#Apply gamma transformation on the image
    def gammatransform(self, gamma, c):
        cv2.destroyAllWindows()
        global image1path
        global transformedimage
        img = cv2.imread(image1path)
        c_value=int(c)
        gamma_value=float(gamma)
        # Perform gamma transorm.
        gammaimg = np.array(c_value * (img /c_value) ** gamma_value, dtype='uint8')
        transformedimage=gammaimg
        # Display original and gamma images
        self.showimage(gammaimg)



#Apply bitplane transformation on the image
    def bitplane(self):
        # Read image into variable img, covert image to Greyscale using 0 flag
        img=cv2.imread(image1path)
        gryimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Iterate over each pixel and change pixel value to binary using np.binary_repr() and store it in a list.
        lst = []
        for i in range(gryimg.shape[0]):
            for j in range(gryimg.shape[1]):
                lst.append(np.binary_repr(gryimg[i][j], width=8))  # width = no. of bits


# We have a list of strings where each string represents binary pixel value. To extract bit planes we need to iterate over the strings and store the characters corresponding to bit planes into lists.
    # Multiply with 2^(n-1) and reshape to reconstruct the bit image.
        eight_bit_img = (np.array([int(i[0]) for i in lst],dtype = np.uint8) * 128).reshape(gryimg.shape[0],gryimg.shape[1])
        seven_bit_img = (np.array([int(i[1]) for i in lst],dtype = np.uint8) * 64).reshape(gryimg.shape[0],gryimg.shape[1])
        six_bit_img = (np.array([int(i[2]) for i in lst],dtype = np.uint8) * 32).reshape(gryimg.shape[0],gryimg.shape[1])
        five_bit_img = (np.array([int(i[3]) for i in lst],dtype = np.uint8) * 16).reshape(gryimg.shape[0],gryimg.shape[1])
        four_bit_img = (np.array([int(i[4]) for i in lst],dtype = np.uint8) * 8).reshape(gryimg.shape[0],gryimg.shape[1])
        three_bit_img = (np.array([int(i[5]) for i in lst],dtype = np.uint8) * 4).reshape(gryimg.shape[0],gryimg.shape[1])
        two_bit_img = (np.array([int(i[6]) for i in lst],dtype = np.uint8) * 2).reshape(gryimg.shape[0],gryimg.shape[1])
        one_bit_img = (np.array([int(i[7]) for i in lst],dtype = np.uint8) * 1).reshape(gryimg.shape[0],gryimg.shape[1])

        # Concatenate these images for ease of display using cv2.hconcat()
        finalr = cv2.hconcat([eight_bit_img, seven_bit_img, six_bit_img, five_bit_img])
        finalv = cv2.hconcat([four_bit_img, three_bit_img, two_bit_img, one_bit_img])

        # Vertically concatenate
        final = cv2.vconcat([finalr, finalv])
        self.showimage(final)


#Log operation dialog box
    def log(self):
        dialog = QDialog()
        dialog.setModal(True)
        text_field = QLineEdit(dialog)
        text_field.setMaxLength(3)
        text_field.setPlaceholderText("ENTER C VALUE")
        text_field.move(30, 10)
        button_transform = QPushButton("Apply Transformation", dialog)
        button_transform.clicked.connect(lambda: self.logtransform(text_field.text()))
        button_transform.move(10, 70)
        dialog.setFixedSize(200, 100)
        dialog.exec_()

#Apply log operations on the image
    def logtransform(self, c):
        cv2.destroyAllWindows()
        c_value=int(c)
        img=cv2.imread(image1path)
        np.seterr(divide='ignore')
        try:
            # Perform log transform
            c_value = c_value / np.log(1 + np.max(img))
            logimg = c_value * (np.log(img + 1))
            logimg = np.array(logimg, dtype=np.uint8)

            # Display original and log images
            self.showimage(logimg)

        except:
            print('Error has been raised')
            # End LogImage Function

#Apply linear transformation on the image
    def linear(self):
        img = cv2.imread(image1path, 0)
        out = 2.0 * img
        out[out > 255] = 255
        out = np.around(out)
        out = out.astype(np.uint8)
        self.showimage(img)

#Thresholding dialog box
    def threshold(self):
        dialog = QDialog()
        dialog.setModal(True)
        text_field = QLineEdit(dialog)
        text_field.setMaxLength(3)
        text_field.setPlaceholderText("ENTER THRESHOLD VALUE")
        text_field.setFixedWidth(180)
        text_field.move(10, 10)
        button_transform = QPushButton("Apply Transformation", dialog)
        button_transform.clicked.connect(lambda: self.thresholding(text_field.text()))
        button_transform.move(10, 70)
        dialog.setFixedSize(200, 100)
        dialog.exec_()

#Apply thresholding transformation on image
    def thresholding(self,value):
        pix_value = int(value)
        img = cv2.imread(image1path,0)
        ret, thresh1 = cv2.threshold(img, pix_value, 255, cv2.THRESH_BINARY)
        self.showimage(thresh1)

#Apply binirization on the image
    def binarization(self):
        img = cv2.imread(image1path, 0)
        ret, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.showimage(otsu)

#Piecewise Dialog box
    def piecewiselinear(self):

        dialog = QDialog()
        dialog.setModal(True)
        text_field = QLineEdit(dialog)
        text_field.setMaxLength(3)
        text_field.setPlaceholderText("R1")
        text_field.move(30, 10)
        text_field.setFixedWidth(20)
        text_field1= QLineEdit(dialog)
        text_field1.setMaxLength(3)
        text_field1.setPlaceholderText("S1")
        text_field1.move(60, 10)
        text_field1.setFixedWidth(20)
        text_field2 = QLineEdit(dialog)
        text_field2.setMaxLength(3)
        text_field2.setPlaceholderText("R2")
        text_field2.move(90, 10)
        text_field2.setFixedWidth(25)
        text_field3 = QLineEdit(dialog)
        text_field3.setMaxLength(3)
        text_field3.setPlaceholderText("S2")
        text_field3.move(120, 10)
        text_field3.setFixedWidth(25)
        button_transform = QPushButton("Apply Transformation", dialog)
        button_transform.clicked.connect(lambda: self.piecewisetransform(text_field.text(),text_field1.text(),text_field2.text(),text_field3.text()))
        button_transform.move(10, 70)
        dialog.setFixedSize(200, 100)
        dialog.exec_()

#Apply piecewise transformation on the image
    def piecewisetransform(self,r1,s1,r2,s2):
        global transformedimage
        img=cv2.imread(image1path)
        r1_value=int(r1)
        s1_value=int(s1)
        r2_value=int(r2)
        s2_value=int(s2)
        pixelvalue=np.vectorize(self.piecewisepixel)
        contraststretching=pixelvalue(img,r1_value,s1_value,r2_value,s2_value)
        plt.imshow(contraststretching.astype('uint8'))
        plt.show()



#Calculate piecewise pixelation for transfomration
    def piecewisepixel(self,img, r1, s1, r2, s2):
        if (0 <= img and img <= r1):
            return (s1 / r1) * img
        elif (r1 < img and img <= r2):
            return ((s2 - s1) / (r2 - r1)) * (img - r1) + s1
        else:
            return ((255 - s2) / (255 - r2)) * (img - r2) + s2

#Set operations dialog box
    def setoperation(self):
        set_operations = ["Union","Intersection","Difference","Complementation"]
        dialog=QDialog()
        dialog.setModal(True)
        combobox=QComboBox(dialog)
        combobox.setGeometry(QtCore.QRect(100, 40, 201, 22))
        combobox.addItems(set_operations)
        button_selectimage2=QPushButton("Select Second Image",dialog)
        button_selectimage2.clicked.connect(self.selectimage2)
        button_transform = QPushButton("Apply Transformation", dialog)
        button_transform.clicked.connect(lambda:self.setoperation_apply(combobox.currentText()))
        button_transform.move(200, 70)
        dialog.setFixedSize(400,100)
        dialog.exec_()

#Apply set operation transformation
    def setoperation_apply(self,combobox):
        cv2.destroyAllWindows()
        img=cv2.imread(image1path)
        img1=cv2.imread(image2path)
        img_resize = cv2.resize(img, (512, 512))
        img1_resize = cv2.resize(img1, (512, 512))
        if (combobox == 'Union'):
            union = img | img1
            self.showimage(union)


        if (combobox == 'Intersection'):
            intersection = img & img1
            self.showimage(intersection)

        if (combobox == 'Difference'):
            difference = img - img1
            self.showimage(difference)


        if (combobox == 'Complementation'):
            complementation = img ^ img1
            self.showimage(complementation)





if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
