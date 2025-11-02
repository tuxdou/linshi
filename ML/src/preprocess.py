import re
from typing import Tuple
from text_unidecode import unidecode
from email.utils import parseaddr

def normalize_name(s):
    if not s:
        return ""
    s = unidecode(s)        
    s = s.lower().strip()  
    s = s.replace(".", " ")
    s = re.sub(r"[^\w\s\-]", "", s, flags=re.UNICODE)  
    s = re.sub(r"\s+", " ", s).strip()  
    return s

def split_name(full):
    full = full.strip()           
    if not full:                 
        return "", ""
    parts = full.split()         
    if len(parts) == 1:          
        return parts[0], ""
    return parts[0], parts[-1] 

def normalize_email(e):
    if not e:
        return "", "", ""

    addr = parseaddr(str(e))[1] or str(e)
    addr = addr.strip().lower()


    if "@" not in addr:
        addr_no_space = addr.replace(" ", "")
        if "@" not in addr_no_space:
            return addr_no_space, addr_no_space, ""

        addr = addr_no_space

    local, domain = addr.split("@", 1)

    if domain in ("gmail.com", "googlemail.com"):
        domain = "gmail.com"
        local = local.split("+", 1)[0].replace(".", "")

    return f"{local}@{domain}", local, domain