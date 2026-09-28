import numpy as np
from itertools import product


def get_song(pau, dur, cpau, cdur):
    """Build a single chirp pattern and return its waveform and pulse count.

    Args:
        pau (int): Pulse pause in samples.
        dur (int): Pulse duration in samples.
        cpau (int): Chirp pause in samples.
        cdur (int): Chirp duration in samples.

    Returns:
        tuple[np.ndarray, int]: Tuple of (chirp waveform, number of pulses).
            The waveform has shape (cdur + cpau, 1).
    """
    cper = max(1, (dur + pau))
    npul = int(np.floor(cdur / cper))
    if npul == 0:
        chirp = np.zeros((cper, 1))
    else:
        pulse = np.concatenate([np.ones((dur, 1)), np.zeros((pau, 1))], axis=0)
        chirp = np.tile(pulse[:, 0], reps=int(npul))[:, np.newaxis]
    return np.concatenate([chirp, np.zeros((int(cdur + cpau - len(chirp)), 1))]), npul


def makeCHIRPstim(
    cpauMax=1000,
    cdurMax=1000,
    cpauMin=20,
    cdurMin=20,
    pdur=20,
    ppau=20,
    Fs=1000,
    num_steps=20,
    cpau=None,
    cdur=None,
    stim_len=5_000,
    spacing="geometric",
):
    """Generate a matrix of chirp stimuli over chirp pause/duration grids.

    Args:
        cpauMax (int, optional): Maximum chirp pause in ms. Defaults to 1000.
        cdurMax (int, optional): Maximum chirp duration in ms. Defaults to 1000.
        cpauMin (int, optional): Minimum chirp pause in ms. Defaults to 20.
        cdurMin (int, optional): Minimum chirp duration in ms. Defaults to 20.
        pdur (int, optional): Pulse duration in ms. Defaults to 20.
        ppau (int, optional): Pulse pause in ms. Defaults to 20.
        Fs (int, optional): Sampling rate in Hz. Defaults to 1000.
        num_steps (int, optional): Number of steps per dimension. Defaults to 20.
        cpau (np.ndarray | None, optional): Chirp pauses in ms. Defaults to None.
        cdur (np.ndarray | None, optional): Chirp durations in ms. Defaults to None.
        stim_len (int, optional): Total stimulus length in samples. Defaults to 5000.
        spacing (str, optional): "linear" or "geometric". Defaults to "geometric".

    Returns:
        tuple[np.ndarray, dict, np.ndarray, np.ndarray]: Tuple of
            (stimulus matrix, stimulus metadata dict, cpau grid, cdur grid).
    """
    dwnSmp = 1000 / Fs
    if cpau is not None and cdur is not None:
        cpau0 = (cpau / dwnSmp).astype(np.uint64)
        cdur0 = (cdur / dwnSmp).astype(np.uint64)
        cpau_cdur = zip(cpau0, cdur0)
        l = max(len(cpau0), len(cdur0))
    else:
        if spacing == "linear":
            cpau0 = (np.linspace(cpauMin, cpauMax, num=num_steps) / dwnSmp).astype(np.uint64)
            cdur0 = (np.linspace(cdurMin, cdurMax, num=num_steps) / dwnSmp).astype(np.uint64)
        else:
            cpau0 = (np.geomspace(cpauMin, cpauMax, num=num_steps) / dwnSmp).astype(np.uint64)
            cdur0 = (np.geomspace(cdurMin, cdurMax, num=num_steps) / dwnSmp).astype(np.uint64)
        cpau_cdur = product(cpau0, cdur0)
        l = len(cpau0) * len(cdur0)
    ppau = int(ppau / dwnSmp)
    pdur = int(pdur / dwnSmp)
    stim_dict = dict()
    stim_dict["cpau"] = np.zeros((l,), dtype=int)
    stim_dict["cdur"] = np.zeros((l,), dtype=int)
    stim_dict["npul"] = np.zeros((l,), dtype=int)
    stim_dict["stim"] = []

    for cnt, (cpau, cdur) in enumerate(cpau_cdur):
        input_stim, npul = get_song(ppau, pdur, cpau, cdur)

        N_chirps = int(np.floor(stim_len / np.shape(input_stim)[0]))
        chirp = input_stim[:, 0]
        stim_dict["npul"][cnt] = npul
        song = np.tile(chirp, reps=N_chirps)[:, np.newaxis]

        # stim_dict['stim'].append(np.concatenate([song,np.zeros((5000 - int(len(song)), 1))]))
        stim_dict["stim"].append(np.concatenate([np.zeros((50, 1)), song, np.zeros((stim_len - int(len(song)), 1))]))
        stim_dict["cpau"][cnt] = cpau
        stim_dict["cdur"][cnt] = cdur
    stim_matrix = np.concatenate(stim_dict["stim"], axis=1)

    stim_dict["CPER"] = stim_dict["cdur"] + stim_dict["cpau"]
    stim_dict["CDC"] = stim_dict["cdur"] / stim_dict["CPER"]
    return stim_matrix, stim_dict, cpau0, cdur0


def makePPFstim(
    ppauMax=80,
    pdurMax=80,
    ppauMin=0,
    pdurMin=0,
    cdur=200,
    cpau=200,
    Fs=1000,
    step=1,
    ppau=None,
    pdur=None,
):
    """Generate a matrix of pulse pair/frequency stimuli over pause/duration grids.

    Args:
        ppauMax (int, optional): Maximum pulse pause in ms. Defaults to 80.
        pdurMax (int, optional): Maximum pulse duration in ms. Defaults to 80.
        ppauMin (int, optional): Minimum pulse pause in ms. Defaults to 0.
        pdurMin (int, optional): Minimum pulse duration in ms. Defaults to 0.
        cdur (int, optional): Chirp duration in ms. Defaults to 200.
        cpau (int, optional): Chirp pause in ms. Defaults to 200.
        Fs (int, optional): Sampling rate in Hz. Defaults to 1000.
        step (int, optional): Step size in ms. Defaults to 1.
        ppau (np.ndarray | None, optional): Pulse pauses in ms. Defaults to None.
        pdur (np.ndarray | None, optional): Pulse durations in ms. Defaults to None.

    Returns:
        tuple[np.ndarray, dict, np.ndarray, np.ndarray]: Tuple of
            (stimulus matrix, stimulus metadata dict, ppau grid, pdur grid).
    """
    dwnSmp = 1000 / Fs
    if ppau is not None and pdur is not None:
        ppau = (ppau / dwnSmp).astype(np.uint64)
        pdur = (pdur / dwnSmp).astype(np.uint64)
        ppau_pdur = zip(ppau, pdur)
    else:
        ppau = (np.arange(ppauMin, ppauMax, step) / dwnSmp).astype(np.uint64)
        pdur = (np.arange(pdurMin, pdurMax, step) / dwnSmp).astype(np.uint64)
        ppau_pdur = product(ppau, pdur)
    cpau = int(cpau / dwnSmp)
    cdur = int(cdur / dwnSmp)

    stim_dict = dict()
    stim_dict["ppau"] = np.zeros((len(ppau) * len(pdur),), dtype=np.uint64)
    stim_dict["pdur"] = np.zeros((len(ppau) * len(pdur),), dtype=np.uint64)
    stim_dict["npul"] = np.zeros((len(ppau) * len(pdur),), dtype=np.uint64)
    stim_dict["stim"] = []
    for cnt, (pau, dur) in enumerate(ppau_pdur):
        pulse = np.concatenate([np.ones((dur, 1)), np.zeros((pau, 1))], axis=0)
        stim_dict["npul"][cnt] = max(1, int(np.floor(cdur / (len(pulse) + 0.000001))))
        chirp = np.tile(pulse[:, 0], reps=int(stim_dict["npul"][cnt]))[:, np.newaxis]
        stim_dict["stim"].append(np.concatenate([np.zeros((50, 1)), chirp, np.zeros((cdur + cpau - len(chirp), 1))]))
        stim_dict["ppau"][cnt] = pau
        stim_dict["pdur"][cnt] = dur

    stim_dict["pper"] = stim_dict["pdur"] + stim_dict["ppau"]
    stim_dict["pdc"] = stim_dict["pdur"] / (stim_dict["pper"] + 0.000001)
    stim_dict["cpau"] = cpau * np.ones_like(stim_dict["ppau"])
    stim_dict["cdur"] = cdur * np.ones_like(stim_dict["ppau"])
    stim_dict["cper"] = stim_dict["pdur"] + stim_dict["ppau"]
    stim_dict["cdc"] = stim_dict["pdur"] / (stim_dict["pper"] + 0.000001)

    stim_matrix = np.concatenate(stim_dict["stim"], axis=1)
    return stim_matrix, stim_dict, ppau, pdur
