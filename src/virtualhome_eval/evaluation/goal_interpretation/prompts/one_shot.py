prompt="""
Your task is to understand natural language goals for a household robot, reason about the object states and relationships, and turn natural language goals into symbolic goals in the given format. The goals include: node goals describing object states, edge goals describing object relationships and action goals describing must-to-do actions in this goal. The input will be the goal's name, the goal's description, relevant objects as well as their current and all possible states, and all possible relationships between objects. The output should be the symbolic version of the goals.


Relevant objects in the scene indicates those objects involved in the action execution initially. It will include the object name, the object initial states, and the object all possible states. It follows the format: object name, id: ...(object id), states: ...(object states), possible states: ...(all possible states). Your proposed object states should be within the following set: CLOSED, OPEN, ON, OFF, SITTING, DIRTY, CLEAN, LYING, PLUGGED_IN, PLUGGED_OUT.


Relevant objects in the scene are:
<object_in_scene>

All possible relationships are the keys of the following dictionary, and the corresponding values are their descriptions:
<relation_types>


Symbolic goals format:
Node goals should be a list indicating the desired ending states of objects. Each goal in the list should be a dictionary with two keys 'name' and 'state'. The value of 'name' is the name of the object, and the value of 'state' is the desired ending state of the target object. For example, [{'name': 'washing_machine', 'state': 'PLUGGED_IN'}, {'name': 'washing_machine', 'state': 'CLOSED'}, {'name': 'washing_machine', 'state': 'ON'}] requires the washing_machine to be PLUGGED_IN, CLOSED, and ON. It can be a valid interpretation of natural language goal: 
Task name: Wash clothes. 
Task description: Washing pants with washing machine
This is because if one wants to wash clothes, the washing machine should be functioning, and thus should be PLUGGED_IN, CLOSED, and ON.

Edge goals is a list of dictionaries indicating the desired relationships between objects. Each goal in the list is a dictionary with three keys 'from_name', and 'relation' and 'to_name'. The value of 'relation' is desired relationship between 'from_name' object to 'to_name' object. The value of 'from_name' and 'to_name' should be an object name. The value of 'relation' should be an relationship. All relations should only be within the following set: ON, INSIDE, BETWEEN, CLOSE, FACING, HOLDS_RH, HOLDS_LH.

Each relation has a fixed set of objects to be its 'to_name' target. Here is a dictionary where keys are 'relation' and corresponding values is its possible set of 'to_name' objects:
<rel_obj_pairs>

Action goals is a list of actions that must be completed in the goals. The number of actions is less than three. If node goals and edge goals are not enough to fully describe the goal, add action goals to describe the goal. Below is a dictionary of possible actions, whose keys are all possible actions and values are corresponding descriptions. When output actions goal list, each action goal should be a dictionary with keys 'action' and 'description'.
<action_space>

Goal name and goal description:
<goal_str>

Now output the symbolic version of the goal. Output in json format, whose keys are 'node goals', 'edge goals', and 'action goals', and values are your output of symbolic node goals, symbolic edge goals, and symbolic action goals, respectively. That is, {'node goals': SYMBOLIC NODE GOALS, 'edge goals': SYMBOLIC EDGE GOALS, 'action goals': SYMBOLIC ACTION GOALS}. Please strictly follow the symbolic goal format.
"""

if __name__ == "__main__":
    pass