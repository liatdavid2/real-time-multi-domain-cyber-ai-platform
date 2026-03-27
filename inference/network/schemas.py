from pydantic import BaseModel
from typing import Optional


class FlowInput(BaseModel):
    dur: Optional[float] = None
    sbytes: Optional[float] = None
    dbytes: Optional[float] = None

    sttl: Optional[int] = None
    dttl: Optional[int] = None

    sloss: Optional[int] = None
    dloss: Optional[int] = None

    sload: Optional[float] = None
    dload: Optional[float] = None

    spkts: Optional[int] = None
    dpkts: Optional[int] = None

    swin: Optional[int] = None
    dwin: Optional[int] = None

    stcpb: Optional[int] = None
    dtcpb: Optional[int] = None

    smeansz: Optional[int] = None
    dmeansz: Optional[int] = None

    trans_depth: Optional[int] = None
    res_bdy_len: Optional[int] = None

    sjit: Optional[float] = None
    djit: Optional[float] = None

    stime: Optional[int] = None
    ltime: Optional[int] = None

    sintpkt: Optional[float] = None
    dintpkt: Optional[float] = None

    tcprtt: Optional[float] = None
    synack: Optional[float] = None
    ackdat: Optional[float] = None

    is_sm_ips_ports: Optional[int] = None
    ct_state_ttl: Optional[int] = None
    ct_flw_http_mthd: Optional[int] = None
    is_ftp_login: Optional[int] = None
    ct_ftp_cmd: Optional[int] = None

    ct_srv_src: Optional[int] = None
    ct_srv_dst: Optional[int] = None

    ct_dst_ltm: Optional[int] = None
    ct_src_ltm: Optional[int] = None

    ct_src_dport_ltm: Optional[int] = None
    ct_dst_sport_ltm: Optional[int] = None
    ct_dst_src_ltm: Optional[int] = None