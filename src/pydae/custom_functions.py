import pkgutil


def load():

    defs_pwm_string   = pkgutil.get_data(__name__, "c_code/pwm.h").decode().replace('\r\n','\n') 
    sources_pwm_string = pkgutil.get_data(__name__, "c_code/pwm.c").decode().replace('\r\n','\n') 

    defs = ''
    sources = ''

    defs += defs_pwm_string
    sources += sources_pwm_string

    return defs,sources
