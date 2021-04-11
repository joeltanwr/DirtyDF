import math
class Latlong:
    """ Supporting class to execute Latlong-related stainers """
    #limited to taking in decimal lat and long
    def __init__(self, lat, long):
        self.lat = lat
        self.long = long
        
    @staticmethod
    def deg_to_dms(deg, type='lat'):
        decimals, number = math.modf(deg)
        d = deg
        m = decimals * 60
        s = (deg - int(d) - int(m) / 60) * 3600.00
        sgn_map = {
            'lat': ('N','S'),
            'long': ('E','W')
        }
        sgn = sgn_map[type][0 if d >= 0 else 1]
        return (abs(d), abs(m), abs(s), sgn)
    
    @staticmethod
    def _round(x, num_decimal):
        if num_decimal == 0:
            return f"{math.floor(x)}"
        else:
            return f"{x:.{num_decimal}f}"

    @staticmethod
    def df_to_latlong(df, lat_idx, long_idx):
        """
        Convert lat and long columns in a pandas dataframe to a single Latlong column. Returns the Latlong column without altering the dataframe.

        Parameters
        ----------
        df : pandas dataframe
            the dataframe to extract Latlong column from
        lat_idx : integer
            the index of the decimal latitude column in the dataframe
        long_idx : integer
            the index of the decimal longitude column in the dataframe

        Returns
        -------
        pd.Series
           the series describing the Latlong column
        """
        return df.apply(lambda x: Latlong(x[lat_idx], x[long_idx]))


    def strflatlong(self, format):
        """
        Convert a latlong object to string based on given user format.

        Parameters
        ----------
        format : string
            the string latlong format to be used.

            Format uses the following notations: %<base><num_decimals><lat / long>
                Base:
                - %D - degree w/ sign (integer (rounded down) if not followed by a number, else decimal)
                - %d - degree w/o sign (integer (rounded down) if not followed by a number, else decimal)
                - %c - N, S, W, or E
                - %m - minutes (integer (rounded down) if not followed by a number, else decimal)
                - %s - seconds (integer (rounded down) if not followed by a number, else decimal)

                Specification for lat/long & number of decimals:
                - %x{n} - x up to n decimals (e.g. %s2 for seconds up to 2 decimals)
                - %xa - to specify x for latitude (e.g. %s2a for latitude seconds up to 2 decimals)
                - %xo - to specify x for longitude (e.g. %m3o for longitude minutes up to 3 decimals)

            There are also special pre-defined format strings which can be used:
                'DMS': Standard DMS format
                'MinDec': Degrees (integer) and Minutes (real number)

        Returns
        -------
        string
            the latlong string format.
        """
        d_lat, m_lat, s_lat, sgn_lat = Latlong.deg_to_dms(self.lat, type='lat')
        d_long, m_long, s_long, sgn_long = Latlong.deg_to_dms(self.long, type='long')
        
        if format == 'DMS':
            format = "%da°%ma\'%s3a%ca, %do°%mo\'%s3o%co"
        elif format == 'MinDec':
            format = "%da°%m3a%ca, %do°%m3o%co"

        #other formats
        output_str = ""
        i = 0
        while i < len(format):
            ch = format[i]
            if ch == '%':
                end_idx = min([x for x in (format[i:].find('a'), format[i:].find('o')) if x > 0]) + i
                ao_type = format[end_idx]
                if end_idx - i == 1: #empty token
                    raise Exception(f"empty token seen at index {i} of format")
            
                if end_idx - i != 2 and not format[i+2: end_idx].isdigit():
                        raise Exception(f"format is not accepted, error caused by following token at" + 
                                f" index {i} to {end_idx+1}: {format[i: end_idx + 1]}")
                
                base_ch = format[i+1] #base character

                if end_idx - i == 2:
                    num_decimal = 0
                else:
                    num_decimal = int(format[i+2: end_idx])
                    
                if base_ch == 'd':
                    if ao_type == 'a':
                        output_str += Latlong._round(d_lat, num_decimal)
                    else:
                        output_str += Latlong._round(d_long, num_decimal)

                elif base_ch == 'D':
                    if (ao_type == 'a' and sgn_lat == 'S') or (ao_type == 'o' and sgn_long == 'W'): #handle sign
                        output_str += "-"
                    if ao_type == 'a':
                        output_str += Latlong._round(d_lat, num_decimal)
                    else:
                        output_str += Latlong._round(d_long, num_decimal)

                elif base_ch == 'c':
                    if ao_type == 'a':
                        output_str += sgn_lat
                    else:
                        output_str += sgn_long

                elif base_ch == 'm':
                    if ao_type == 'a':
                        output_str += Latlong._round(m_lat, num_decimal)
                    else:
                        output_str += Latlong._round(m_long, num_decimal)
                
                elif base_ch == 's':
                    if ao_type == 'a':
                        output_str += Latlong._round(s_lat, num_decimal)
                    else:
                        output_str += Latlong._round(s_long, num_decimal)

                i = end_idx + 1    
            
            else: #not % token
                output_str += format[i]
                i += 1
        
        return output_str