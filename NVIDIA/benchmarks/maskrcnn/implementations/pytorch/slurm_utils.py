import subprocess
from ast import literal_eval

def parse_aux_nodes(aux_nodes):
    aux_nodes = aux_nodes.split(",")[1:-1]
    parsed_nodes = []
    for node in aux_nodes:
        if "-" in node:
            low = literal_eval(node.split("-")[0])
            high = literal_eval(node.split("-")[1])
            node_range = list(range(low, high+1))
            parsed_nodes.extend(node_range)
        else:
            parsed_nodes.append(int(node))
    return parsed_nodes

def get_all_nodes():
    slurm_info = subprocess.check_output("sinfo").decode("utf-8")
    if "idle" not in slurm_info:
        print("No idle nodes")
        return
    slurm_info = slurm_info.split("idle")[1].strip()
    slurm_list = ["ip{}".format(i) for i in slurm_info.split("ip")[1:]]
    nodes = []
    for i in slurm_list:
        main_node = i.split("[")[0]
        aux_nodes = i.split("[")
        if len(aux_nodes)>1:
            aux_nodes = "[{}".format(aux_nodes[1])
            if aux_nodes.endswith(','):
                aux_nodes = aux_nodes[:-1]
            aux_nodes = parse_aux_nodes(aux_nodes)
            for j in aux_nodes:
                nodes.append("{}{}".format(main_node, j))
        else:
            if main_node.endswith(','):
                main_node = main_node[:-1]
            nodes.append(main_node)
    return nodes

def get_available_gpu_nodes():
    nodes = get_all_nodes()
    if nodes==None:
        return
    with open("/home/ubuntu/master_host") as hostfile:
        host_names = hostfile.readlines()
        host_names = [i.strip() for i in host_names]
    gpu_nodes = set(nodes).intersection(set(host_names))
    gpu_nodes = list(gpu_nodes)
    gpu_nodes = [i.replace("ip-","").replace("-",".") for i in gpu_nodes]
    return gpu_nodes

