# DESDEOv2
A whole refactoring of DESDEO for a simpler, more scalable and distributable
framework for developing and experimenting with interactive multiobjective
optimization methods.

## About
This project is meant to be a refactoring (actually a rewrite) of the DESDEO
framework. 

DESDEO stands for **DE**cision **S**upport for computationally **DE**manding
**O**ptimization problems.  It is meant to serve as a framework to develop and
experiment with interactive multiobjective optimization methods.

## Basic program structure (WIP, RFC)
### Reasons for this refactoring
The basic layout of the original DESDEO framework is confusing and is not
encouraging for a new user. One of the main goals of this refactoring, is to
follow a [KISS](https://en.wikipedia.org/wiki/KISS_principle)-mentality. The
proposed structure is a modular one, with one master module (Manager) meant to
work as a message broker between the other modules. Each module should, in
principle, work as an autonomous unit. This will allow for better scaling,
testing and distributed/threaded computing later in the development path of
DESDEO. The original version of DESDEO is not scalable and the different modules
are intertwined together, which makes it hard to add and/or modify new/existing
features.

The original version is also full of bugs and testing is practically
non-existent. This will make further development risky and frustrating, and lead
to many more bugs. It is therefore not wise to continue building on top of the
original version, at least not in its' current state.

Refactoring the original version of DESDEO does not seem wise. The project is not
big, but the aforementioned intertwined class hierarchy makes refactoring a nightmare.
It will probably take just as long, if not longer, to refactor the old version than
rewriting it. Hence, this project.

### General structure
A proposed simple modular structure is presented in the picture below. It is
quite general in the terms used, but it is supposed to serve as a sketch of the
program's final structure.

![Diagram of the structure of DESDEOv2](https://github.com/gialmisi/DESDEOv2/blob/master/assests/DESDEOv2_structure.png "Concept structure of DESDEOv2")

## Development
*Disclaimer*: Everything is under consideration at the moment and the project is
in a planning stage.

### Goals
The very first goal of this project is to bring it, functionality wise, on par
with the original version.

### Issues
For comments, suggestions and issues, create issue tickets. You can also email
me at gialmisi (at) jyu (dot) fi.

## See also
[DESDEO project page](https://desdeo.it.jyu.fi/)  
[Original DESDEO framework](https://github.com/industrial-optimization-group/DESDEO)



