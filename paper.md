---
title: 'ASGarD: Adaptive Sparse Grid Discretization'
tags:
  - C++
  - fusion
  - plasma
  - advection
  - high dimensional
authors:
  - name: Ed D'Azevedo
    orcid: 0000-0000-0000-0000
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Wael Elwasif
    orcid: 0000-0000-0000-0000
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: David L. Green
    orcid: 0000-0003-3107-1170
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Hao Lau
    orcid: 0000-0000-0000-0000
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Graham Lopez
    orcid: 0000-0000-0000-0000
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Adam McDaniel
    orcid: 0000-0000-0000-0000
    affiliation: "1, 4" # (Multiple affiliations must be quoted)
  - name: Benjamin T. McDaniel
    orcid: 0000-0000-0000-0000
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Lin Mu
    orcid: 0000-0000-0000-0000
    affiliation: "1, 3" # (Multiple affiliations must be quoted)
  - name: Timothy Younkin
    orcid: 0000-0000-0000-0000
    affiliation: 1 # (Multiple affiliations must be quoted)
affiliations:
 - name: Oak Ridge National Laboratory
   index: 1
 - name: University of Tennessee Knoxville
   index: 2
 - name: University of Georgia
   index: 3
 - name: South Doyle High School, Knoxville, TN
   index: 4
date: 12 June 2019
bibliography: paper.bib
---
# Statement of Need

Many areas of science exhibit physical processes which are well described by high dimensional partial differential equations (PDEs), e.g., the 4D, 5D and 6D models describing magnetized fusion plasmas [@Juno:2017], or the ... models describing quantum mechanical interactions of several bodies [@...]. In such problems, the so called "curse of dimensionality" whereby the number of degrees of freedom (or unknowns) required to be solved for scales as $N^D$ where $N$ is the number of grid points in any given dimension. A simple, albeit naive, 6D example with $N=1000$ grid points in each dimension would require more than an exabyte of memory just for store the solution vector, not to mention forming the matrix required to advance such a system in time. While there are methods to simulate such high-dimensional systems, they are mostly based on Monte-Carlo methods which are based on a statistical sampling such that the resulting solutions are noisy. Since the noise in such methods can only be reduced at a rate proportional to $\sqrt{N_p}$ where $N_p$ is the number of Monte-Carlo samples, there is a need for continuum, or grid / mesh based methods for high-dimensional problems which both do not suffer from noise, but which additionally bypass the curse of dimenstionality. Here we present a simulation framework which provides such a method. 

# Summary

The forces on stars, galaxies, and dark matter under external gravitational
fields lead to the dynamical evolution of structures in the universe. The orbits
of these bodies are therefore key to understanding the formation, history, and
future state of galaxies. The field of "galactic dynamics," which aims to model
the gravitating components of galaxies to study their structure and evolution,
is now well-established, commonly taught, and frequently used in astronomy.
Aside from toy problems and demonstrations, the majority of problems require
efficient numerical tools, many of which require the same base code (e.g., for
performing numerical orbit integration).

``Gala`` is an Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for ``Gala`` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. ``Gala`` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the ``Astropy`` package [@astropy] (``astropy.units`` and
``astropy.coordinates``).

``Gala`` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in ``Gala`` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike.

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$


# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this: ![Example figure.](figure.png)

# Acknowledgements

This research used resources of the Oak Ridge Leadership Computing Facility at the Oak Ridge National Laboratory, which is supported by the Office of Science of the U.S. Department of Energy under Contract No. DE-AC05-00OR22725.

This research used resources of the National Energy Research Scientific Computing Center (NERSC), a U.S. Department of Energy Office of Science User Facility operated under Contract No. DE-AC02-05CH11231.

# References
