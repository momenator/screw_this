import PySimpleGUI as sg

layout = []

objects = [['productCode', 'length', 'quantity'],
           ['productCode2', 'length2', 'quantity2']]


def gui(objectsall):
    objectList = objectsall
    for index, item in enumerate(objectList):
        layout.append([sg.Text(item[0])])
        layout.append([sg.Text(item[1])])
        layout.append([sg.Text(item[2])])
    return sg.Window('Window Title', layout)


# All the stuff inside your window. This is the PSG magic code compactor...
# layout = [[sg.Text('Some text on Row 1')],
 #         [sg.Text('Enter something on Row 2'), sg.InputText()],
#          [sg.OK(), sg.Cancel()]]


# Create the Window
# Event Loop to process "events"
window = gui(objects)
while True:
    event, values = window.Read()
    if event in (None, 'Cancel'):
        break

window.Close()
