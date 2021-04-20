from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from einops import rearrange
import torch.nn.functional as F
import torch

def parallel_process(array, function, n_jobs=4, use_kwargs=False, front_num=1):
    """
        A parallel version of the map function with a progress bar. 

        Args:
            array (array-like): An array to iterate over.
            function (function): A python function to apply to the elements of array
            n_jobs (int, default=16): The number of cores to use
            use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of 
                keyword arguments to function 
            front_num (int, default=3): The number of iterations to run serially before kicking off the parallel job. 
                Useful for catching bugs
        Returns:
            [function(array[0]), function(array[1]), ...]
    """
    #We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]
    #If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs==1:
        return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[front_num:])]
    #Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        #Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
        #Print out the progress as tasks complete
        for f in tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    #Get the results from the futures. 
    for i, future in tqdm(enumerate(futures)):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return front + out


def interpolate_output(output, in_freq, out_freq):
    # output [Time Cls]
    scale = out_freq // in_freq
    length = output.size()[0] # time length
    out_length = scale * (length - 1) + 1 # make sure each sample point is aligned
    output = F.interpolate(rearrange(output, '(1 T) C -> 1 C T'), out_length, mode='linear', align_corners=True)
    output = rearrange(output, '1 C T -> (1 T) C')
    # print(length, out_length, output.size()[0])
    return output

def correlation(output, labels, dim = 0):
    # assumed shape [S 15]
    # implements Pearson's Correlation
    x = output
    y = labels

    vx = x - torch.mean(x, dim=dim, keepdim=True) # mean along the temporal axis [S 15] - [1 15]
    vy = y - torch.mean(y, dim=dim, keepdim=True)

    cor = torch.sum(vx * vy, dim=dim) / (torch.sqrt(torch.sum(vx ** 2, dim=dim) * torch.sum(vy ** 2, dim=dim)) + 1e-6) # [15]
    mean_cor = torch.mean(cor)
    return mean_cor, cor