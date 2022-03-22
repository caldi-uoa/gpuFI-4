# gpuFI-4: A Microarchitecture-Level Framework for Assessing the Cross-Layer Resilience of Nvidia GPUs

gpuFI-4 is a detailed microarchitecture-level fault injection framework to assess the cross-layer vulnerability of hardware structures and entire GPU chips for single and multiple bit faults, built on top of the state-of-the-art simulator [GPGPU-Sim](https://github.com/gpgpu-sim/gpgpu-sim_distribution). The target hardware structures that gpuFI-4 can perform error injection campaigns are the register file, the shared memory, the L1 data and texture caches and the L2 cache.

## REFERENCE
If you use gpuFI-4 for your research, please cite:

>D. Sartzetakis, G. Papadimitriou, and D. Gizopoulos, “gpuFI-4: A Microarchitecture-Level Framework for Assessing the Cross-Layer Resilience of Nvidia GPUs", IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS 2022), Singapore, May 22-24 2022.

The full ISPASS 2022 paper for gpuFI-4 can be found [here](http://cal.di.uoa.gr/wp-content/uploads/2022/3/gpuFI-4_ISPASS_2022.pdf).

## INSTALLING and BUILDING gpuFI-4

gpuFI-4 is developed on top of GPGPU-Sim 4.0 and several input parameters have 
been created for this purpose which are passed through the gpgpusim.config file 
to the simulator. The installation and the building process is identical to [GPGPU-Sim](https://github.com/gpgpu-sim/gpgpu-sim_distribution).

If your purpose is to use gpuFI-4 to evaluate the fault effects of a CUDA application using
PTXPLUS and not PTX then make sure that you are compiling GPGPU-Sim and the application with CUDA 4.2 or less
as PTXPLUS currently only supports sm_1x. 

Here is the message from the GPGPU-Sim developers during the setup_environment that
explains it thoroughly:

>INFO - If you only care about PTX execution, ignore this message. GPGPU-Sim supports PTX execution in modern CUDA.
>If you want to run PTXPLUS (sm_1x SASS) with a modern card configuration - set the environment variable
>$PTXAS_CUDA_INSTALL_PATH to point a CUDA version compabible with your card configurations (i.e. 8+ for PASCAL, 9+ for VOLTA etc..)
>For example: "export $PTXAS_CUDA_INSTALL_PATH=/usr/local/cuda-9.1"
>
>The following text describes why:
>If you are using PTXPLUS, only sm_1x is supported and it requires that the app and simulator binaries are compiled in CUDA 4.2 or less.
>The simulator requires it since CUDA headers desribe struct sizes in the exec which change from gen to gen.
>The apps require 4.2 because new versions of CUDA tools have dropped parsing support for generating sm_1x
>When running using modern config (i.e. volta) and PTXPLUS with CUDA 4.2, the $PTXAS_CUDA_INSTALL_PATH env variable is required to get proper register >usage
>(and hence occupancy) using a version of CUDA that knows the register usage on the real card.

## PREPARING and PROFILING

There are two steps that needs to be done before starting the injection campaigns.
The first step is to make some minor modifications to the CUDA application and the
second step is to set all the necessary parameters in the campaign.sh accordingly.


### Step 1: CUDA application preparation

gpuFI-4 relies its evaluation process on the evaluation of the application itself. As a result, 
the applications should be slightly modified to compare the results of the GPU part execution 
with either a predefined result file (taken from a fault-free execution) or the results that come 
from the CPU “golden” reference execution and print a custom message (rather unique to the GPGPU-Sim output)
in the standard output. The custom messages of your choice upon a successful or a failed execution will 
need to be set in SUCCESS_MSG, FAILED_MSG parameters of the campaign.sh 


### Step 2: Campaign script preparation

The campaign.sh script (the file with example values already uploaded) requires several 
parameters to be configured before the injection campaigns are performed. We can differentiate
these parameters into four abstract groups. The first group contains one-time parameters.
The second contains parameters that need to be initialized once per GPGPU card and are necessary
to define values that describe some of the hardware structures. In the third group, there are
parameters that need to be initialized every time we analyze the vulnerability of a new CUDA
application or single kernel. Let’s call these groups: one time, per GPGPU card, per
kernel/application and per injection campaign parameters respectively.  
To get the values for some of the parameters in the "per kernel/application" group requires a fault-free execution of the CUDA application in GPGPU-Sim. You can run a fult-free execution with profile=3.  

**One-time parameters**
- **CONFIG_FILE:** This is the GPGPU-Sim configuration file where the new input parameters must be defined. The configuration files that we used in our paper are already
uploaded under configs/tested-cfgs/{SM75_RTX2060, SM7_QV100, SM3_KEPLER_TITAN}. They have the new parameters defined and PTXPLUS enabled.
- **RUNS:** This is how many executions the campaign is going to run. For example, if we set the campaign.sh to inject a bit flip on one register and we have RUNS=3000 then 3000 application executions will be performed by injecting a bit flip on a random register on each run.
- **BATCH:** To make the campaigns faster we provided it with some kind of parallelism. Specifically, #BATCH number of executions run in parallel until all are finished before starting the next batch. The default value of this parameter is the number of processors (or virtual cores if hyper-threading is supported) minus one core so the system will not hang.
- **TMP_FILE:** This is a file that contains the GPGPU-Sim execution default output along with the CUDA application output.
- **TMP_DIR:** This is the directory where the CONFIG_FILE and GPGPU-Sim output (TMP_FILE) files are saved for each execution. In fact, roundup(RUNS/BATCH) number of TMP_DIR directories will be created appended with an identifier. For example, if we have RUNS=10, BATCH=5, TMP_DIR=logs and TMP_FILE=tmp then logs1 and logs2 directories will be created where each one contains the files {gpgpusim.config1,gpgpusim.config2,...,gpgpusim.config5} and {tmp1, tmp2,...,tmp5}.
- **CACHE_LOGS_DIR:** This is a directory where logs are saved for all the executions when we run injection campaigns on caches. The information that is saved is the cache line that the fault was injected and the exact bit that was flipped.

**Per GPGPU card parameters**
- **L1D_SIZE_BITS:** This is the total size in bits of the L1 data cache per SM. Tag bits should be included.
- **L1T_SIZE_BITS:** Same as the L1D_SIZE_BITS but for the texture cache.
- **L2_SIZE_BITS:** This is the total size in bits of the L2 cache. Tag bits should be included here as well. 

**Per kernel/application parameters**
- **CUDA_UUT:** The CUDA application command.
- **CYCLES:** The total cycles that the application took on a fault-free execution, meaning without any fault injections. GPGPU-Sim is deterministic and thus each fault free execution of the same program with the same inputs gives the same number of clock cycles.
- **profile=1:** This will run the application once without any fault injections and output the cycles for each kernel’s invocation at TMP_FILE during the last cycle of the application, which we can use as input to initialize the CYCLES_FILE parameter. The two previous parameters CUDA_UUT and CYCLES are required for this profiling to work.
- **profile=3:** This will run the application once without any fault injections. Is the same as profile=1 but without any computations that makes the fault-free application execution slower. 
- **CYCLES_FILE:** This is a file that contains all the cycles one by one per line that will be used for our injections. A random cycle from this file is chosen before every execution. With this file, our framework is capable of performing injections on specific cycles like on a kernel invocation, on all the invocations of a kernel, or the whole application.
- **MAX_REGISTERS_USED:** This is the maximum number of registers that a kernel uses per thread. 
- **SHADER_USED:** This is the SIMT cores that a kernel uses.
- **SUCCESS_MSG, FAILED_MSG:** This is the success and failure message respectively that an application prints after its own evaluation.
- **TIMEOUT_VAL:** This is the timeout of an execution which is useful in case the execution of an application hangs. The format is the one needed for the timeout command in Linux.
- **LMEM_SIZE_BITS:** This is the size in bits that a kernel uses for the local memory per thread.
- **SMEM_SIZE_BITS:** This is the size in bits that a kernel uses for the shared memory per CTA.

**Per injection campaign parameters**
- **profile=0:** By setting the profile value to 0, the profiling procedure will be disabled and the actual injection campaigns will be executed. 
- **components_to_flip:** This is the hardware structure on which the injections will be applied. The value that describes a specific structure can be found within the campaign script. If a user wishes, can also perform injections on multiple components per execution by inserting more than one component value with a colon as a delimiter. For example, with components_to_flip=0:2 injections will be done on both register file and shared memory at the same execution.   
- **register_rand_n:** This is the number of the register that the transient faults will be injected. In gpuFI-4 we are not targeting specific registers by name, so the value can be a number between 1 to MAX_REGISTERS_USED. Again this parameter can be crafted with more registers using a colon as a delimiter in case we want to inject the same fault on multiple registers and the same practice has been applied to all the parameters that end with ‘_n’. Furthermore, a ‘_rand’ on a parameter’s name indicates that on each execution the value will be changed randomly between some boundaries.
- **reg_bitflip_rand_n:** This is the specific bit that will be flipped.
- **per_warp:** If activated with the value of 1 then #register_rand_n registers will have their #reg_bitflip_rand_n bits flipped on every thread of an active warp. Otherwise, one running thread only will be affected. 
- **shared_mem_bitflip_rand_n:** Same as reg_bitflip_rand_n but for the shared memory. This will randomly choose, in every execution, value(s) between 1 to SMEM_SIZE_BITS.
- **blocks:** This is on how many running CTAs, hence shared memories, to inject #shared_mem_bitflip_rand_n bit flips.
- **l1d_cache_bitflip_rand_n:** Same as reg_bitflip_rand_n but for the L1 data cache. This will randomly choose, in every execution, value(s) between 1 to L1D_SIZE_BITS.
- **l1d_shader_rand_n:** This is in which running SIMT core, hence L1 data cache, to inject shared_mem_bitflip_rand_n bit flips.
- **l1t_cache_bitflip_rand_n, l1t_shader_rand_n:** Same like L1 data cache but they are used for the texture cache.
- **l2_cache_bitflip_rand_n:** Same as reg_bitflip_rand_n but for L2 cache. This will randomly choose, in every execution, value(s) between 1 to L2_SIZE_BITS.


## INJECTION CAMPAIGN EXECUTION AND RESULTS

After setting up everything described on the previous section, the fault injection campaign can be easily executed
by simply running the campaign.sh script. The script eventually will go on a loop (until it reaches #RUNS cycles),
where each cycle will modify the framework’s new parameters at gpgpusim.config file before executing the application.

After completion of every batch of fault injections, a parser post-processes the output of the experiments one
by one and accumulates the results. The final results will be printed when all the batches have finished and
the script has quit. The parser classifies the fault effects of each experiment as Masked, Silent Data Corruption (SDC),
or Detected Unrecoverable Error (DUE).
- **Masked:** Faults in this category let the application run until the end and the result is identical to that of a fault-free execution. 
- **Silent Data Corruption (SDC):** The behavior of an application with these types of faults is the same as with masked faults but the application’s result is incorrect. These faults are difficult to identify as they occur without any indication that a fault has been recorded (an abnormal event such as an exception, etc.).
- **Detected Unrecoverable Error (DUE):** In this case, an error is recorded and the application reaches an abnormal state without the ability to recover.

We additionally use the term “Performance” as a fault effect which is nothing but a Masked fault effect where the total cycles of the application are different from the fault-free execution.
