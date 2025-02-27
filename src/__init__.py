# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.toolshed import BundleAPI


# Subclass from chimerax.core.toolshed.BundleAPI and
# override the method for registering commands,
# inheriting all other methods from the base class.
class _segmentMap(BundleAPI):

    api_version = 1     # start_tool called with BundleInfo and
                        # CommandInfo instance instead of command name
                        # (when api_version==0)

    # Override method
    @staticmethod
    def start_tool(session, bi, ti):
        # bi is an instance of chimerax.core.toolshed.BundleInfo
        # ci is an instance of chimerax.core.toolshed.CommandInfo

        # This method is called once for each command listed
        # in bundle_info.xml.  Since we list two commands,
        # we expect two calls to this method.

        # We check the name of the command, which should match
        # one of the ones listed in bundle_info.xml
        # (without the leading and trailing whitespace),
        # and import the function to call and its argument
        # description from the ``tool`` module.
        if ti.name == 'locscalesurfer':
            from . import tool 
            return tool.SegmentMapTool(session, ti.name)
        else:
            raise ValueError("@#@# Unknown command: " + ti.command)
        
    @staticmethod
    def get_class(class_name):
        if class_name == "SegmentMapTool":
            from . import tool
            return tool.SegmentMapTool
        else:
            raise ValueError("Unknown class #@: " + class_name)


# Create the ``bundle_api`` object that ChimeraX expects.
bundle_api = _segmentMap()