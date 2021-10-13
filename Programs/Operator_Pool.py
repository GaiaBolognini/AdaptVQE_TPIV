
import pennylane as qml
from pennylane import numpy as np
from pennylane import qchem


#operator pool
def operator_1(params):
    qml.RX(np.pi/2, wires=[0])
    qml.CNOT(wires=[3,2])
    qml.CNOT(wires=[2,1])
    qml.CNOT(wires=[1,0])
    qml.RZ(params, wires=[0])
    qml.CNOT(wires=[1, 0])
    qml.CNOT(wires=[2,1])
    qml.CNOT(wires=[3,2])
    qml.RX(-np.pi/2, wires=[0])


def operator_2(params):
    qml.RX(np.pi/2, wires = [1])
    qml.CNOT(wires = [3,2])
    qml.CNOT(wires=[2,1])
    qml.RZ(params, wires=[1])
    qml.CNOT(wires=[2,1])
    qml.CNOT(wires=[3, 2])
    qml.RX(-np.pi / 2, wires=[1])


def operator_3(params):
    qml.RX(np.pi/2, wires=[2])
    qml.CNOT(wires=[3,2])
    qml.RZ(params, wires=[2])
    qml.CNOT(wires=[3,2])
    qml.RX(-np.pi/2, wires=[2])


def operator_4(params):
    qml.RX(np.pi/2, wires=[1])
    qml.CNOT(wires=[3,1])
    qml.RZ(params, wires=[1])
    qml.CNOT(wires=[3,1])
    qml.RX(-np.pi/2, wires=[1])

def operator_5(params):
    qml.RX(np.pi/2, wires=[2])
    qml.RZ(params, wires=[2])
    qml.RX(-np.pi/2, wires=[2])
