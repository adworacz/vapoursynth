Python Reference
================

VapourSynth is separated into a core library and a Python module. This section
explains how the core library is exposed through Python and some of the
special things unique to Python scripting, such as slicing and output.


VapourSynth Structure
#####################

To use the VapourSynth library you must first retrieve the Core object, using
get_core(). This core may then load plugins, which all end up in their own unit,
or namespace, so to say, to avoid naming conflicts in the contained functions.
For this reason you call a plugin function with *core.unit.Function()*.

All arguments to functions have names that are lowercase and all function names
are CamelCase. Unit names are also lowercase and usually short. This is good to
remember. If you do not like CamelCase for function names you can pass
*accept_lowercase=True* to get_core().

Slicing
#######

The VideoNode class (always called "clip" in practice) supports the full
range of indexing and slicing operations in Python. If you do perform a slicing
operation on a clip, you will get a new clip back with the desired frames.
Here are some examples to illustrate::

   # ret will be a one frame clip containing the 6th frame
   ret = clip[5]
   # ret will contain frames 6 to 9 (unlike Trim, the end value of python slicing is not inclusive)
   ret = clip[6:10]

   # Select even numbered frames
   ret = clip[::2]
   # Select odd numbered frames
   ret = clip[1::2]

   # Negative step is also allowed, so this reverses a clip
   ret = clip[::-1]

   # It may all be combined at once to confuse people, just like normal Python lists
   ret = clip[-400:-800:-5]

Output
######

The normal way of specifying the clip(s) to output is to call
*clip.set_output()*. All standard VapourSynth components only use output
index 0, but other tools may use something different.
There are also other variables that can be set to control how a format is
output. For example, setting *enable_v210=True* changes the packing of the
YUV422P10 format to one that is common in professional software (like Adobe
products).
An example on how to get v210 output::

   some_clip = core.resize.Bicubic(clip, format=vs.YUV422P10)
   some_clip.set_output()
   enable_v210 = True

Raw Access to Frame Data
########################

The VideoFrame class simply contains one picture and all the metadata
associated with it. It is possible to access the raw data using ctypes and
some persistence. The three relevant functions are *get_read_ptr(plane)*,
*get_write_ptr(plane)*, and *get_stride(plane)*, all of which take the plane
to access as an argument. Accessing the data is a bit trickier as
*get_read_ptr()* and *get_write_ptr()* only return a pointer. To get a frame
simply call *get_frame(n)* on a clip.

Classes and Functions
#####################
.. py:function:: get_core([threads = 0, add_cache = True, accept_lowercase = False])

   Get the singleton Core object. If it is the first time the function is called,
   the Core will be instantiated with the given options. If the Core has already
   been instantiated, all options are ignored. Setting *threads* to a value
   greater than zero overrides the autodetection.

.. py:function:: set_message_handler(handler_func)

   Sets a function to handle all debug output and fatal errors. The function should have the form *handler(level, message)*,
   where level corresponds to the vapoursynth.mt constants. Passing *None* restores the default handler, which prints to stderr.
   
.. py:function:: get_output([index = 0])

   Get a previously set output node. Throws an error if the index hasn't been
   set.

.. py:function:: clear_output([index = 0])

   Clears a clip previously set for output.

.. py:function:: clear_outputs()

   Clears all clips set for output in the current environment.

.. py:class:: Core

   The *Core* class uses a singleton pattern. Use *get_core()* to obtain an
   instance. All loaded plugins are exposed as attributes of the core object.
   These attributes in turn hold the functions contained in the plugin.
   Use *get_plugins()* to obtain a full list of all currently loaded plugins
   you may call this way.
   
   .. py:attribute:: num_threads
      
      The number of concurrent threads used by the core. Can be set to change the number. Setting to a value less than one makes it default to the number of hardware threads.
      
   .. py:attribute:: add_cache
   
      For debugging purposes only. When set to *False* no caches will be automatically inserted between filters.
      
   .. py:attribute:: accept_lowercase
   
      When set to *True* function name lookups in the core are case insensitive. Don't distribute scripts that need it to be set.
      
   .. py:attribute:: max_cache_size
   
      Set the upper framebuffer cache size after which memory is aggressively
      freed. The value is in megabytes.

   .. py:method:: set_max_cache_size(mb)

      An alias for setting *max_cache_size*. Kept for compatibility with older scripts.

   .. py:method:: get_plugins()

      Returns a dict containing all loaded plugins and their functions.

   .. py:method:: list_functions()

      Works similar to *get_plugins()* but returns a human-readable string.

   .. py:method:: register_format(color_family, sample_type, bits_per_sample, subsampling_w, subsampling_h)

      Register a new Format object or obtain a reference to an existing one if
      it has already been registered. Invalid formats throw an exception.

   .. py:method:: get_format(id)

      Retrieve a Format object corresponding to the specified id. Returns None if there is no format with that *id*.

   .. py:method:: version()

      Returns version information as a string.
      
   .. py:method:: version_number()

      Returns the core version as a number.

.. py:class:: VideoNode

   Represents a video clip. The class itself supports indexing and slicing to
   perform trim, reverse and selectevery operations. Several operators are also
   defined for the VideoNode class: addition appends clips and multiplication
   repeats them. Note that slicing and indexing always return a new VideoNode
   object and not a VideoFrame.

   .. py:attribute:: format

      A Format object describing the frame data. If the format can change
      between frames, this value is None.

   .. py:attribute:: width

      The width of the video. This value will be 0 if the width and height can
      change between frames.

   .. py:attribute:: height

      The height of the video. This value will be 0 if the width and height can
      change between frames.

   .. py:attribute:: num_frames

      The number of frames in the clip. This value will be 0 if the total number
      of frames is unknown or infinite.

   .. py:attribute:: fps_num

      The numerator of the framerate. If the clip has variable framerate, the
      value will be 0.

   .. py:attribute:: fps_den

      The denominator of the framerate. If the clip has variable framerate, the
      value will be 0.

   .. py:attribute:: flags

      Special flags set for this clip. This attribute should normally be
      ignored.

   .. py:method:: get_frame(n)

      Returns a VideoFrame from position n.

   .. py:method:: set_output(index = 0)

      Set the clip to be accessible for output. This is the standard way to
      specify which clip(s) to output. All VapourSynth tools (vsvfw, vsfs,
      vspipe) use the clip in *index* 0.

   .. py:method:: output(fileobj[, y4m = False, prefetch = 0, progress_update = None])
 
      Write the whole clip to the specified file handle. It is possible to pipe to stdout by specifying *sys.stdout* as the file.
      YUV4MPEG2 headers will be added when *y4m* is true.
      The current progress can be reported by passing a callback function of the form *func(current_frame, total_frames)* to *progress_update*.
      The *prefetch* argument is only for debugging purposes and should never need to be changed.
      
.. py:class:: VideoFrame

      This class represents a video frame and all metadata attached to it.

   .. py:attribute:: format

      A Format object describing the frame data.

   .. py:attribute:: width

      The width of the frame.

   .. py:attribute:: height

      The height of the frame.

   .. py:attribute:: readonly

      If *readonly* is True, the frame data and properties cannot be modified.

   .. py:attribute:: props

      This attribute holds all the frame's properties mapped as sub-attributes.

   .. py:method:: copy()

      Returns a writable copy of the frame.

   .. py:method:: get_read_ptr(plane)

      Returns a pointer to the raw frame data. The data may not be modified.
      
   .. py:method:: get_read_array(plane)

      Returns a memoryview of the frame data that's only valid as long as the VideoFrame object exists. The data may not be modified.

   .. py:method:: get_write_ptr(plane)

      Returns a pointer to the raw frame data. It may be modified using ctypes
      or some other similar python package.
      
   .. py:method:: get_write_array(plane)

      Returns a memoryview of the frame data that's only valid as long as the VideoFrame object exists.

   .. py:method:: get_stride(plane)

      Returns the stride between lines in a *plane*.

.. py:class:: Format

   This class represents all information needed to describe a frame format. It
   holds the general color type, subsampling, number of planes and so on.
   The names map directly to the C API so consult it for more detailed
   information.

   .. py:attribute:: id

      A unique *id* identifying the format.

   .. py:attribute:: name

      A human readable name of the format.

   .. py:attribute:: color_family

      Which group of colorspaces the format describes.

   .. py:attribute:: sample_type

      If the format is integer or floating point based.

   .. py:attribute:: bits_per_sample

      How many bits are used to store one sample in one plane.

   .. py:attribute:: bytes_per_sample

      The actual storage is padded up to 2^n bytes for efficiency.

   .. py:attribute:: subsampling_w

      The subsampling for the second and third plane in the horizontal
      direction.

   .. py:attribute:: subsampling_h

      The subsampling for the second and third plane in the vertical direction.

   .. py:attribute:: num_planes

      The number of planes the format has.

.. py:class:: Plugin

   Plugin is a class that represents a loaded plugin and its namespace.

   .. py:method:: get_functions()

      Returns a dict containing all the functions in the plugin. You can access
      it by calling *core.std.get_functions()*. Replace *std* with the namespace
      of the plugin you want to query.

   .. py:method:: list_functions()

      Works similar to *get_functions()* but returns a human-readable string.
      
.. py:class:: Function

   Function is a simple wrapper class for a function provided by a VapourSynth plugin.
   Its main purpose is to be called and nothing else.
   
.. py:class:: Func

   Func is a simple wrapper class for VapourSynth VSFunc objects.
   Its main purpose is to be called and manage reference counting.

.. py:exception:: Error

   The standard exception class. This exception is thrown on most errors
   encountered in VapourSynth.

Color Family Constants
######################

The color family constants describe groups of formats and the basic way their
color information is stored. You should be familiar with all of them apart from
maybe *YCOCG* and *COMPAT*. The latter is a special junk category for non-planar
formats. These are the declared constants in the module::

   RGB
   YUV
   GRAY
   YCOCG
   COMPAT

Format Constants
################

Format constants exactly describe a format. All common and even more uncommon
formats have handy constants predefined so in practice no one should really
need to register one of their own. These values are mostly used by the resizers
to specify which format to convert to. The naming system is quite simple. First
the color family, then the subsampling (only YUV has it) and after that how many
bits per sample in one plane. The exception to this rule is RGB, which has the
bits for all 3 planes added together. The long list of values::

   GRAY8
   GRAY16
   GRAYH
   GRAYS

   YUV420P8
   YUV422P8
   YUV444P8
   YUV410P8
   YUV411P8
   YUV440P8

   YUV420P9
   YUV422P9
   YUV444P9

   YUV420P10
   YUV422P10
   YUV444P10

   YUV420P16
   YUV422P16
   YUV444P16

   YUV444PH
   YUV444PS

   RGB24
   RGB27
   RGB30
   RGB48

   RGBH
   RGBS

   COMPATBGR32
   COMPATYUY2

Sample Type Constants
#####################

::

   INTEGER
   FLOAT
