FROM quay.io/fenicsproject/stable:current

RUN export PYVISTA_OFF_SCREEN=true
RUN apt-get update
RUN apt-get install -y --no-install-recommends libgl1-mesa-dev xvfb
RUN pip install ipykernel -U --user
RUN pip install numpy -U --user
#RUN pip3 install pyvista matplotlib
