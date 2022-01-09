# NPSCProject-Code

README FOR NADITH
  Code isn't interactive - modules are all present to test different algorithms,
  but have to manually adjust code to switch between them.
  
  If necessary, I can streamline this to make it more user friendly, but I was
  focussed on just getting results in time for end of year.
  
  MAIN
    Sets bounds (global variables) of testing.
    Calls other modules and does some testing of modules.
    
  INPUT_IMAGE
    Imports the .tif file

  PREPROCESSING
    Does preprocessing that is common to all algorithms (i.e., doesn't make
    adjustments that different algs specifically require, but does split
    into centre & neighbourhood, etc.)

  IMAGEPROCESSING
    Split into X, Y training set, and ML algorithms.

  OUTPUT_IMAGE
    Self-evident. Used to get example images. It was going to generate the red/blue
    images like in your paper to evaluate effectiveness of each algorithm
    numerically, but never finished this. The code remnants are there, but aren't called.

fin.
