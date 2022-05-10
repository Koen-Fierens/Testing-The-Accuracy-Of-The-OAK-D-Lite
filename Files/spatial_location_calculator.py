#!/usr/bin/env python3
import ossd

import cv2
import depthai as dai
from statistics import mean
from openpyxl import Workbook
import xlsxwriter
import pandas as pd
from openpyxl import load_workbook

recording = False
depthList = []

#VERANDER EXCELFILE VOOR VERSCHILLENDE AFSTANDEN (ExcelSheetNumber niet handmatig aanpassen!)
#VOLG PATRONEN CORRECT VOOR JUISTE SHEET!
ExcelMeasurementFolder = "C:/Users/KoenF/OneDrive/Documenten/School/Jaar1/Semester2/Bachelorproef/Metingen/Metingen/"
ExcelFile = "5.5m.xlsx"
ExcelSheetNumber = 1

def placeMeasurementsExcel(Filename,Filepath, Sheetname,df):
    try:
        xlsx_file = pd.ExcelFile(Filepath+Filename)
    except:
        xlsxwriter.Workbook(Filepath+Filename)
        xlsx_file = pd.ExcelFile(Filepath+Filename)
    writer = pd.ExcelWriter(Filepath+Filename, engine='openpyxl')
    IsSheetThereAlready = False
    for sheet in xlsx_file.sheet_names:
        if sheet == Sheetname:
            df2 = xlsx_file.parse(sheet)
            df2.to_excel(writer,sheet_name= sheet, index=False)
            df.to_excel(writer,sheet_name= sheet, startrow=len(df2)+1, index=False, header=None)
            IsSheetThereAlready = True
        else:
            df2 = xlsx_file.parse(sheet)
            df2.to_excel(writer,sheet_name= sheet, index=False)
    if IsSheetThereAlready is False:

        df.to_excel(writer,sheet_name = Sheetname, index=False)
    writer.save()
    return

def toggleRecording():
    global ExcelSheetNumber
    global ExcelFile
    global depthList
    global recording
    if recording:
        recording = False
        print("You stopped recording!\n")
        data = pd.DataFrame({"Metingen": depthList})
        placeMeasurementsExcel(ExcelFile, ExcelMeasurementFolder,f"Patroon{ExcelSheetNumber}", data)
        ExcelSheetNumber = ExcelSheetNumber + 1
        depthList.clear()
    else:
        recording = True
        print("You started recording!\n")


stepSize = 0.01

newConfig = False

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)

xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutSpatialData = pipeline.create(dai.node.XLinkOut)
xinSpatialCalcConfig = pipeline.create(dai.node.XLinkIn)

xoutDepth.setStreamName("depth")
xoutSpatialData.setStreamName("spatialData")
xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

# Properties
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

lrcheck = False
subpixel = False

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setLeftRightCheck(lrcheck)
stereo.setSubpixel(subpixel)

# Config
topLeft = dai.Point2f(0.4, 0.4)  # x,y
bottomRight = dai.Point2f(0.41, 0.41)  # x,y

config = dai.SpatialLocationCalculatorConfigData()  # object dat de rechthoek bevat alsook de diepte waarin gemeten kan worden
config.depthThresholds.lowerThreshold = 100  # tot hoe dicht mag gemeten worden
config.depthThresholds.upperThreshold = 10000  # tot hoe ver mag gemeten worden
config.roi = dai.Rect(topLeft, bottomRight)  # maak een rechthoek

spatialLocationCalculator.inputConfig.setWaitForMessage(False)
spatialLocationCalculator.initialConfig.addROI(config)

# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

spatialLocationCalculator.passthroughDepth.link(xoutDepth.input)
stereo.depth.link(spatialLocationCalculator.inputDepth)

spatialLocationCalculator.out.link(xoutSpatialData.input)
xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    # Output queue will be used to get the depth frames from the outputs defined above
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)  # plek waar de diepte waardes instaan
    spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=4,
                                             blocking=False)  # plek waar nieuwe dieptes berekent moeten worden
    spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")

    color = (255, 0, 0)

    print("Use ZQSD keys to move ROI!\n")
    print("Use R key to toggle recording!\n")
    print("Use E key to exit!\n")

    while True:
        inDepth = depthQueue.get()  # Blocking call, will wait until a new data has arrived

        depthFrame = inDepth.getFrame()  # depthFrame values are in millimeters, depthframe krijgt 1 frame binnen die berekent moet worden

        depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF,
                                        cv2.CV_8UC1)  # pixels aanpassen voor contrast en stoppen het in depthFramecolor
        depthFrameColor = cv2.equalizeHist(
            depthFrameColor)  # histogram equalization, nog een manier voor contrast aan te passen
        depthFrameColor = cv2.applyColorMap(depthFrameColor,
                                            cv2.COLORMAP_HOT)  # we gaan de grijswaarden naar roodwaarden brengen

        spatialData = spatialCalcQueue.get().getSpatialLocations()  # berekenen van afstanden in "spatialcalcqueue" en in var spatialdata stoppen
        for depthData in spatialData:  # we gaan over alle data in spatialdata
            roi = depthData.config.roi  # roi is Region Of Interest. we maken deze van depthData
            roi = roi.denormalize(width=depthFrameColor.shape[1],
                                  height=depthFrameColor.shape[0])  # depthFrameColor.shape[0], depthFrameColor.shape[0]
            xmin = int(roi.topLeft().x)
            ymin = int(roi.topLeft().y)
            xmax = int(roi.bottomRight().x)
            ymax = int(roi.bottomRight().y)

            # depthMin = depthData.depthMin
            # depthMax = depthData.depthMax

            fontType = cv2.FONT_HERSHEY_TRIPLEX
            cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, 1)
            cv2.putText(depthFrameColor, f"X: {int(depthData.spatialCoordinates.x)} mm", (xmin + 10, ymin + 20),
                        fontType, 0.5, (255, 0, 0))
            cv2.putText(depthFrameColor, f"Y: {int(depthData.spatialCoordinates.y)} mm", (xmin + 10, ymin + 35),
                        fontType, 0.5, (255, 0, 0))
            cv2.putText(depthFrameColor, f"Z: {int(depthData.spatialCoordinates.z)} mm", (xmin + 10, ymin + 50),
                        fontType, 0.5, (255, 0, 0))
        # Show the recording window

        if recording == True:
            if len(depthList) >= 100:
                toggleRecording()
            else:
                depthList.append(int(depthData.spatialCoordinates.z))

        # Show the frame
        cv2.imshow("depth", depthFrameColor)

        key = cv2.waitKey(1)
        if key == ord('e'):
            break
        elif key == ord('z'):
            if topLeft.y - stepSize >= 0:
                topLeft.y -= stepSize
                bottomRight.y -= stepSize
                newConfig = True
        elif key == ord('q'):
            if topLeft.x - stepSize >= 0:
                topLeft.x -= stepSize
                bottomRight.x -= stepSize
                newConfig = True
        elif key == ord('s'):
            if bottomRight.y + stepSize <= 1:
                topLeft.y += stepSize
                bottomRight.y += stepSize
                newConfig = True
        elif key == ord('d'):
            if bottomRight.x + stepSize <= 1:
                topLeft.x += stepSize
                bottomRight.x += stepSize
                newConfig = True
        elif key == ord('r'):
            toggleRecording()

        if newConfig:
            config.roi = dai.Rect(topLeft, bottomRight)
            config.calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.AVERAGE
            cfg = dai.SpatialLocationCalculatorConfig()
            cfg.addROI(config)
            spatialCalcConfigInQueue.send(cfg)
            newConfig = False
