import sys, math
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

def draw():
    glClear(GL_COLOR_BUFFER_BIT)
    glLoadIdentity()  # Reset the modelview matrix

    glutWireTeapot(0.7)
    glFlush()

def main():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB)
    glutInitWindowSize(750, 700)
    glutInitWindowPosition(100, 100)
    glutCreateWindow("Wavefunction Simulation")

    glutDisplayFunc(draw)

    glutMainLoop()

if __name__ == "__main__":
    main()
