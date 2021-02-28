from tkinter import *

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from Examples import Examples
from Neural_Network import Neural_Network
from Point import Point

root = Tk()

root.geometry('800x800')
root.title("BackPropagation")


def translate(center, angle, arm_length):
    return Point(center.x + arm_length * np.sin(angle), center.y - arm_length * np.cos(angle))


def train_clicked():
    for i in range(10000):
        nn.train(x_train, y_train)
    varOut.set("Trained!")
    err = nn.errors
    # plt.plot(range(len(err)), err)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(range(len(err)), range(len(err)), err)
    plt.savefig('error.pdf')
    plt.close()


# Updating arm and pointer position
def mouse_moved(event):
    varOut.set("")
    global arm
    global forearm
    global oval
    canvas.after(0, canvas.delete, oval)
    canvas.after(0, canvas.delete, arm)
    canvas.after(0, canvas.delete, forearm)
    x = event.x
    y = event.y
    oval = canvas.create_oval(x - 6, y - 6, x + 6, y + 6, fill="green")
    out = nn.forward([((x - min) / max) * 0.8 + 0.1, ((y - min) / max) * 0.8 + 0.1])
    alpha = ((out[0] - 0.1) / 0.8) * 3.14
    beta = ((out[1] - 0.1) / 0.8) * 3.14
    p = translate(Point(400, 400), alpha, 200)
    arm = canvas.create_line(400, 400, p.x, p.y)
    w = translate(p, 3.14 - beta + alpha, 200)
    forearm = canvas.create_line(p.x, p.y, w.x, w.y)


canvas = Canvas(root, height=800, width=800, bg="white")
button_learn = Button(canvas, text="Learn", command=train_clicked).place(x=0, y=720, width=100, height=50)
varOut = StringVar()
label = Label(canvas, textvariable=varOut, font=(None, 25)).place(x=0, y=670, width=100, height=50)
varOut.set("")

# Generate the examples
ex = Examples()
e = ex.generate(1000)
# Draw examples plot
fig, ax = plt.subplots()
ax.axis('equal')
for (x, y) in e[0]:
    plt.scatter(x, y, marker='o')
plt.savefig('generated_points.pdf')
plt.close()
# Scaling training data
min = np.min(e[0])
max = np.max(e[0])
x_train = ((np.array(e[0]) - min) / max) * 0.8 + 0.1
y_train = np.array(e[1]) / 3.14 * 0.8 + 0.1
nn = Neural_Network()

# Creating a robot
oval = canvas.create_oval(x - 2, y - 2, x + 2, y - 2, fill="white")
arm = canvas.create_line(400, 400, 400, 200)
forearm = canvas.create_line(400, 200, 600, 200)
canvas.create_rectangle(200, 350, 400, 700, fill="grey")
canvas.create_rectangle(200, 700, 250, 800, fill="grey")
canvas.create_rectangle(350, 700, 400, 800, fill="grey")
canvas.create_rectangle(280, 300, 320, 350, fill="grey")
canvas.create_rectangle(250, 200, 350, 300, fill="grey")
canvas.create_rectangle(260, 210, 280, 230, fill="blue")
canvas.create_rectangle(320, 210, 340, 230, fill="blue")
canvas.create_rectangle(270, 260, 330, 280, fill="white")
canvas.create_line(290, 260, 290, 280, fill="black")
canvas.create_line(310, 260, 310, 280, fill="black")
canvas.create_line(200, 400, 100, 600)
canvas.create_line(100, 600, 0, 400)

canvas.bind("<Motion>", mouse_moved)

canvas.pack()


root.mainloop()
