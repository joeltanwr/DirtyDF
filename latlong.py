import math
class Latlong:
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

    def strfgeo(self, format):
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