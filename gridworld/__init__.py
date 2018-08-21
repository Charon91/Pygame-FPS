from gym.envs.registration import register

#register(
#    id='LabyrinthBasic-v0',
#    entry_point='gymlabyrinth.envs:LabyrinthBase'
#)

register(
    id='Gridworld4RoomsDiscrete-v0',
    entry_point='gridworld.envs:GridworldFourRooms'
)

register(
    id='Gridworld4RoomsDiscrete360-v0',
    entry_point='gridworld.envs:GridworldFourRooms360'
)

register(
    id='Gridworld4RoomsContinuous-v0',
    entry_point='gridworld.envs:GridworldFourRoomsContinuous'
)

register(
    id='Gridworld4RoomsContinuous360-v0',
    entry_point='gridworld.envs:GridworldFourRoomsContinuous360'
)
