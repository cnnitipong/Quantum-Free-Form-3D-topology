# Third-party notices

FreeTO-Python is a Python port and extension of other people's work. Their licences are kept here as they require.

## FreeTO (MATLAB) - MIT License

The numerical core (mesh generation, SIMP/SEMDOT, filters, smooth-edge projection, STL output) and the example
STL files in `examples/STLs/` (not `examples/STLs/generated/`, which `examples/make_examples.py` creates) are ported from
or taken from FreeTO, https://github.com/ooibhadode/FreeTO
(O. Ibhadode, Y.-F. Fu, A. Qureshi, Advances in Engineering Software 198 (2024) 103790).

    MIT License

    Copyright (c) 2024 CADmaniac

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

## intriangulation / voxelise (MATLAB File Exchange) - BSD 3-Clause License

`freeto/inside.py` re-implements, in vectorised form, the point-in-triangulation test of `intriangulation.m`
(J. Korsawe, File Exchange 43381), which is based on `voxelise` (A. H. Aitkenhead).

    Copyright (c) 2016, Johannes Korsawe
    Copyright (c) 2013, Adam H. Aitkenhead
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.
    * Neither the name of the copyright holder nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Bundled JavaScript libraries (web app, `webapp/static/vendor/`)

- three.js (core, OrbitControls, STLLoader) - MIT License, Copyright 2010-2026 Three.js Authors; licence header kept in the files. https://github.com/mrdoob/three.js
- Chart.js v4.5.1 - MIT License, (c) 2025 Chart.js Contributors; licence header kept in the file. https://github.com/chartjs/Chart.js

## Method references

MMA (`freeto/optimizers.py`) is an independent implementation of the published method of K. Svanberg (1987, 2002);
no MMA source code is included.
