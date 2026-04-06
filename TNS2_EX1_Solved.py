#the connection matrix function that was already defined for me
import numpy as np
import matplotlib.pyplot as plt
def connection_matrix(N):
    """
    Compute the connection matrix and its principal eigenvalues/eigenvectors.

    Params:
        N (int): Size of the matrix.

    Returns:
        tuple: (M, E1, E2, E3, E4, lambda1, lambda2, lambda3, lambda4)
    """
    lambda_ = 0.8
    theta = np.linspace(-np.pi, np.pi, N + 1)
    X, Y = np.meshgrid(theta[:-1], theta[:-1])

    # Fancy code to make distant neighbours positive
    M = lambda_ * np.cos(X - Y) / (np.pi ** 2)

    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eig(M)

    # Sort eigenvalues and eigenvectors in ascending order
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    return M, eigvecs[:, -1], eigvecs[:, -2], eigvecs[:, -3], eigvecs[:, -4], eigvals[-1], eigvals[-2], eigvals[-3], eigvals[-4]

#CALL THE CONNECTION MATRIX FIRST 
N = 20 #NO. OF NEURONS
M, E1, E2, E3, E4, lambda1, lambda2, lambda3, lambda4 = connection_matrix(N)
e1r = E1.real #Value of eigenvector 1 at each neuron and .real ensures no imaginary number
e2r = E2.real
lam1r = lambda1.real
lam2r = lambda2.real
lam3r = lambda3.real
lam4r = lambda4.real

print("M shape:", M.shape)            # should say (20, 20)
print("e1r length:", len(e1r))        # should say 20
print("lam1r:", round(lam1r, 4))      # should say ~0.81
print("lam2r:", round(lam2r, 4))      # should say ~0.81

#step 1: generate several input vectors h_i  
#each h_i stimulates a different adjacent pair of neurons 
#the scalar value h is chosen randomly between [1 and 3] each time 

num_inputs = 5 #number of input vectors
#now we put these input vectors into a list or a box basically 
h_list = []
#below we start a loop. range(num_inputs) = range(5) generates the numbers 0, 1, 2, 3, 4. 
# The variable i takes each value in turn. Everything indented underneath this line runs once per value of i.
for i in range(num_inputs):
    h_i = np.zeros(N) #We start fresh each time through the loop. Every h_i begins as all zeros, and then we place the nonzero values in.
    h_val = np.random.uniform(1, 3) #random value between 1 and 3
    position = i * 2 #this gives us pair positions : 0,2,4,6,8 and calculates which pair of neurons to stimulate 
    #so position 0 is 0x2, position 1 is 1x2, position 2 is 2x2 and so on
    """

    These two lines place the scalar h_val into two adjacent positions in h_i.
    `h_i[position]` accesses the element at index `position` and sets it to h_val.
    `h_i[position + 1]` accesses the next element and sets it to the same h_val.
    """
    h_i[position] = h_val #first of the adjacent pair of neurons
    h_i[position + 1] = h_val #second pair of adjacent neurons
    #so after this h_i looks like: when i=0 [h, h, 0, 0, 0, ...], when i=1 [0, 0, h, h, 0, ...] so the second position, the second pair is stimulated
    #so we insert our h_val in the position of that pair 
    h_list.append(h_i) #add to the list 

#np.random.uniform(1, 3) picks one random decimal number between 1 and 3. For example: 2.341, or 1.087, or 2.899. This is the scalar value — 
# the strength of the input for this particular h_i. It is different every time the loop runs because random.uniform picks a new number each call.

#what can we expect to see after we make the list of input vectos?
# h_list[0] = [h, h, 0, 0, 0, ...]   neurons 1 and 2
# h_list[1] = [0, 0, h, h, 0, ...]   neurons 3 and 4
# h_list[2] = [0, 0, 0, 0, h, h,...] neurons 5 and 6
# h_list[3] = [0, 0, 0, 0, 0, 0, h, h, ...] neurons 7 and 8
# h_list[4] = [0, 0, 0, 0, 0, 0, 0, 0, h, h, ...] neurons 9 and 10

#TASK A - PART 1: Plot e1, e2 and several h_i input vectors vs node index 

node_index = np.arange(1, N+1) #python always excludes the last stop value so if i did (1, N+1) it would include N giving me 20 neurons and not 19

#plt.subplots() creates a figure andn one set of axes that are stored in two variables : 
# fig which is the drawing paper and ax which is the diagram on fig

fig, ax = plt.subplots(figsize = (12,5)) 
#fig size is the size which is 12 inches wide by 5 inches tall, when i want two plots i would use fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax.step(node_index, e1r, label = 'e1', where = 'mid', color = 'darkseagreen', linewidth =2)
ax.step(node_index, e2r, label = 'e2', where = 'mid', color = 'cadetblue', linewidth =2)
colors = ['#888888', '#aaaaaa', '#555555', '#333333', '#bbbbbb'] #this is a hex color code, they are all shades of grey 
"""
what the parameters mean:
ax.step() draws a staircase plot. 
node_index — the x values. [1, 2, 3, ..., 20]. Which neuron on the horizontal axis.
e1r — the y values. The value of eigenvector e1 at each neuron. .real already applied so no imaginary parts.
label='e1' — the name that appears in the legend box.
where='mid' — centres the flat step on each neuron position. Without this, steps would be left-aligned and the plot would look shifted.
color='green' — the colour of the line.
linewidth=2 — the thickness of the line. 2 is slightly thicker than default

"""
for i, h_i in enumerate(h_list):
    """ i is the position number which could be 0,1,2,3,4 and h_i is the actual vector in that position
    enumerate is useful because it gives me an output containing both i and h_i, else i would have to write h_list[i] 
    to access each vector and that is tedious, this is just easier """

    ax.step(node_index, h_i,
            label=f'h{i + 1} (neurons {2*i + 1}, {2*i + 2})',
            where='mid',
            color=colors[i],
            linestyle='--',
            alpha=0.8)
#this is a similar step plot as the one we learnt with one input vector h but now we have multiple input vectors h_i
#label is the name that appears in the legend box, 
#The f before the quote lets you put variables directly inside curly braces. So when i=0: label becomes 'h1 (neurons 1,2)'. When i=1: 'h2 (neurons 3,4)'. 
#And so on. The i+1 is because i starts at 0 but we want to label from 1.
#so it will show h1, h2, h3, h4, h5 and also which neurons are stimulated by each h_i
#color is the colour of the line, we use the colors list to assign a different shade
#linestyle is the style of the line, we use dashed lines for the h_i vectors
#alpha is the transparency of the line, we set it to 0.8 for the h_i vectors

ax.set_xlabel('Node index k')
ax.set_ylabel('Value') #y axis is comprised of the eigenvectore components and input strengths 
ax.set_title('Task A - eigenvectors and input vectors vs node index')
ax.axhline(0, color = 'lightgray', linewidth=0.5) #axhline draws horizontal line across the whole plot, 0 means it draws it at y=0 and this is 
#reference line that helps me see which neuron values are positive and negative , linewidth 0.5 makes this reference line thin so it does not overshadow
ax.legend(loc='upper right')
plt.tight_layout() #adjusts spacing so that labels and titles fit nicely and don't overlap or extend 
plt.show()

#TASK A : PART 2 : PROJECTIN ONTO E1-E2 PLANE 

# Project the input vector x onto the plane spanned by e1 and e2
    # This is done by taking the dot product of x with e1 and e2

def project_onto_plane(x):
    return np.dot(e1r, x), np.dot(e2r, x) #first term is x coord, second is y coord

""" 
- def means defining a function, it's like teaching python a new word of the day, whenever we call project_onto_plane, we are using that new word
later, when we write project_onto_plane(some vector), it runs the two lines and does the action that the definition describes to it
(the meaning of the word we taught it), it's like it uses that new word it just learnt in a sentence. 

- x is any input vector that we want to project, we use x as a placeholder because later we use this function to project h_i
np.dot(e1r, x) = e1*x which is the x coordinate in the plane, and np.dot(e2r, x) = e2*y which is the y coordinate in the plane.

- the function returns both numbers as a tuple. A tuple is a collection of values that are ordered and cannot be changed
 In this case, it returns (proj_e1, proj_e2) as the coordinates in the e1-e2 plane. The values will be packaged together,
 unpacking them would mean using (*proj_e1, *proj_e2) and this would give us the individual x and y coordinates. """

#new figure and axis for the plot
fig, ax = plt.subplots(figsize=(6,6))

#we project the eigenvectors onto themselves in the plane, should be (1,0) and (0,1) for e1 and e2 respectively

p_e1 = project_onto_plane(e1r) #this gives us the coordinates of e1 in the e1-e2 plane, which is (proj_e1, proj_e2)
p_e2 = project_onto_plane(e2r) #this gives us the coordinates of e2 in the e1-e2 plane, which is (proj_e1, proj_e2) 
 
ax.scatter(*p_e1, label = 'e1', color='darkseagreen', zorder=5, s=130, marker='.')
ax.scatter(*p_e2, label = 'e2', color='cadetblue', zorder=5, s=130, marker='.')

"""
ax.scatter draws individual points in the plot 
- the * operator unpacks the x and y coordinates from the tuples p_e1 and p_e2
- zorder controls the drawing order of the points, higher values are drawn on top
- s controls the size of the points
- marker controls the shape of the points """

for i, h_i in enumerate(h_list):
    p_h_i = project_onto_plane(h_i) #this gives us the coordinates of h_i in the e1-e2 plane, which is (proj_e1, proj_e2)
    ax.scatter(*p_h_i, label = f'h{i+1}', color=colors[i], zorder=5, s=100, marker='o')
    ax.annotate('', xy=p_h_i, xytext=(0,0), arrowprops=dict(arrowstyle='->', color=colors[i], lw=1.5), zorder=4)

"""
- we loop over all five input vectors. For each one:
p_h_i = project_onto_plane(h_i) - projects h_i onto the e1-e2 plane and gives us one (x,y) coordinate. 

-- ax.scatter(*p_h_i, label = f'h{i+1}', color=colors[i], zorder=5, s=100, marker='o') - draws the point for h_i in the plot

-- ax.annotate('', xy=p_h_i, xytext=(0,0), arrowprops=dict(arrowstyle='->', color=colors[i], lw=1.5), zorder=4) - draws an arrow from the origin 
to the point p_h_i.

Here: '' no text label on the arrow itself, xytext=(0,0) means the arrow starts from the origin (0,0), xy=p_h_i means 
that the arrow ends here.

- arrowprops controls the properties of the arrow i want, so i can customize it (color, size, style) """ 

#for each h_i scatter produces a dot and annotate produces an arrow for my plot 

#how to add the x and y axis reference lines in order to plot our points: 

ax.axhline(0, color='lightgray', linewidth=0.5) #reference line for x axis at x=0
ax.axvline(0, color='lightgray', linewidth=0.5) #reference line for y axis at y=0

ax.set_xlabel('Projection onto e1') #e1 * x
ax.set_ylabel('Projection onto e2') #e2 * x
ax.set_title('Task A - Projection of input vectors onto e1-e2 plane')
ax.legend(loc='upper right')
plt.tight_layout()
plt.show()

#TASK B: Compute and visualise several input vectors h, together with their respective steady state solutions v_ss_exact and v_ss_approx in both ways
#as a function of node index and as a point in the e1-e2 plane 

#we get three different representations for each input vector h_i:
#1. The original vector h_i as a function of node index (the input itself)
#2. The steady state solution v_ss_exact as a function of node index 
#3. The steady state solution v_ss_approx as a function of node index 

#In the e1-e2 plane, we can represent each vector as a point based on its projections onto the eigenvectors e1 and e2.

#TO COMPUTE : 
""" we need to calculate v_ss_exact which is (I-M)*v_ss = h. 
this is a linear system of equations where I is an identity matrix and M is the connection matrix of size 20x20, b=h is the input vector
and we want to find x which is our exact steady state solution . 
- np.linalg.solve(A,b) solves the equation A*x = b for x. I give it A and b, and it returns the solution x which is v_ss_exact.
- I=np.eye(N) creates identity matrix of size NxN.

Computing v_ss_approx is taking the dot product of (e1r, h)/(1-lam1r)*e1r (multiply with e1 cause we project it onto e1) + np.dot(e2r, h)/(1-lam2r)*e2r
#use backslashes to continue the expression across multiple lines
"""
#---------------------------Task B-------------------------------------

#for each input vector h_i, we need to compute v_ss_exact and v_ss_approx and visualize it 

#step 1: setup identity matrix with np.eye which we need for the exact steady state formula
I = np.eye(N) 

#we define our projection function as before 
def project_onto_plane(x):
    return np.dot(e1r, x), np.dot(e2r, x) #first term is x coord, second is y coord

#we now compute steady states for all our input vectors 
#we create two lists to hold both types of SS values 

v_ss_exact_list = []
v_ss_approx_list = []

for h_i in h_list:
    #h_list has all my input vectors from task A, we loop over all of them and compute both SS
    #exat steady state: solve (I-M)*v = h for v
    v_e = np.linalg.solve(I - M, h_i) #v-e is the exact steady state solution for input vector h_i
    v_ss_exact_list.append(v_e) #add to the list of exact steady states
    #np.linalg.solve(A,b) finds x such that A*x = b 
    #here A=I-M, b = h_i and x = v_exact_ss

    #approximate steady state: we use two eigenvectors not all of them , also the formula says that too
    v_a = (np.dot(e1r, h_i) / (1 - lam1r)) * e1r + (np.dot(e2r, h_i) / (1 - lam2r)) * e2r
    v_ss_approx_list.append(v_a) #add to the list of approximate steady states
    #np.dot(e1r, h_i) / (1 - lam1r) gives the alignment of h_i with e1
    #dividing by (1-lam1r) is the amplification factor 
    #multiplying by e1 points the result back along e1 
    #same for the e2 part of the formula then we add 

#PLOTTING TASK B : one subplot per input vector, all in one column 
# each subplot shows both steady states and the input vector vs node index 

fig, axes = plt.subplots(num_inputs, 1, figsize=(12, 3*num_inputs)) 
#num_inputs in rows and 1 column
#figsize is the canvas, we now have 5 plot areas, one per h_i
node_index = np.arange(1, N+1) #1-20 neurons for x axis 

for i in range(num_inputs):
    ax = axes[i] #select the i-th subplot where i=0 is the first subplot and so on
    ax.step(node_index, h_list[i], label=f'h{i+1}', where='mid', color='gray', linestyle='--', alpha=0.8)
    #plot the input vector as a dashed grey staircase
    #h_list[i] picks the i-th input vector from our list 

    ax.step(node_index, v_ss_exact_list[i], label='v_ss_exact', where='mid', color='steelblue', linewidth=2)
    ax.step(node_index, v_ss_approx_list[i], label='v_ss_approx', where='mid', color='salmon', linewidth=2) 

    ax.axhline(0, color = 'lightgray', linewidth = 0.5) #xaxis
    ax.set_ylabel('Activity')
    ax.set_title(f'h{i+1}: neurons {i*2+1} and {i*2+2} stimulated')
    ax.legend(loc='upper right', fontsize=8)
    
ax.set_xlabel('Neuron index')
plt.suptitle('Task B - exact vs approx SS', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95]) #rect = [left, bottom, right, top], it tells tightlayout to leave space for the title by leaving 2% of the top
#empty
plt.show() #always leave the plt show unindented else it loops five times for the same plot and won't show

#--------visualization of projection og e1-e2 plane
#all vectors shown as points in one single plot 
#each input vector has matching steady state vectors

fig, ax = plt.subplots(figsize=(7,7))
colors_b = ['steelblue', 'salmon','lightgreen','purple','orange']
#five colors per input and same color for their steady states 

for i in range(num_inputs):

    p_h      = project_onto_plane(h_list[i])
    p_exact  = project_onto_plane(v_ss_exact_list[i])
    p_approx = project_onto_plane(v_ss_approx_list[i])
    # project each vector into the e1-e2 plane
    # each gives one (x, y) point

    col = colors_b[i]
    # pick this input's colour

    # draw arrow from origin to h_i
    ax.annotate('', xy=p_h, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=col,
                                lw=1.5, linestyle='dashed'))
    ax.scatter(*p_h, s=80, color=col, marker='o',
               label=f'h{i+1}')
    # circle marker for input vectors

    # draw arrow from origin to v_exact
    ax.annotate('', xy=p_exact, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=col, lw=2))
    ax.scatter(*p_exact, s=100, color=col, marker='s')
    # square marker for exact steady state
    # marker='s' means square
    # draw arrow from origin to v_approx
    ax.annotate('', xy=p_approx, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=col,
                                lw=1.5, linestyle='dotted'))
    ax.scatter(*p_approx, s=80, color=col, marker='x')
    # x marker for approximate steady state

# add reference lines through origin
ax.axhline(0, color='lightgray', linewidth=0.8)
ax.axvline(0, color='lightgray', linewidth=0.8)

ax.set_xlabel('Component along e1')
ax.set_ylabel('Component along e2')
ax.set_title('Task B — steady states in e1-e2 plane')

# add a manual legend explaining the markers
# because automatic legend gets messy with many colours
from matplotlib.lines import Line2D
legend_markers = [
    Line2D([0], [0], marker='o', color='gray',
           label='input h', markersize=8, linestyle='None'),
    Line2D([0], [0], marker='s', color='gray',
           label='v_ss exact', markersize=8, linestyle='None'),
    Line2D([0], [0], marker='x', color='gray',
           label='v_ss approx', markersize=8, linestyle='None'),
]
#legends to show which color belongs to which input 
legend_colors = [
    Line2D([0], [0], color=colors_b[i], lw=3, label=f'h{i+1} (neurons{i*2+1},{i*2+2})')
    for i in range(num_inputs)
]
""" The above is a list comprehension which is a compact way of creating a for loop, it creates
one Line2D entry per input vector and each entry has the correct colour and label"""

legend1 =ax.legend(handles=legend_markers, loc='upper right')
legend2 = ax.legend(handles=legend_colors, loc='upper left')
ax.add_artist(legend1) #add the first legend to the plot so that it does not get overwritten by the second one!!!!

# handles= lets you pass a custom list of legend items
# instead of using whatever matplotlib collected automatically

plt.tight_layout()
plt.show()

#TASK C - Simulate the time evolution of the system of several input vectors h_i. Start at v=0 and iteratively compute subsequent activity vectors. 
#Proceed from time 0 to time 20 in steps of delta_t = 0.5. Plot the trajectory from v=0 to v_ss in the e1-e2 plane.

#the activity vectors are not static like our input vectors that are localized and appear as a time step because only a pair of neurons get stimulated.
#the activity vectors are the representation of amplification of the input 

#we have two equations, to calculate current equilibrium where v_eq = M*v_i + h, where v_i is an activity vector. We also compute the exponential 
#relaxation to v_ss with v_i + 1 = v_eq + (v_i-v_eq)*e⁻ᵈᵉˡᵗᵃ_ᵗ
#we compute that e^⁻delta t where t is 0.5, so we need to calculate e^(-0.5) and the answer is 0.607 
#we plot the time evolution of the activity pattern of all five input vectors in the e1-e2 plane where the steady state lives from v_0 to v_ss

#set up time parameters 
dt = 0.5
T = 20 #total simulation time 
steps = int(T/dt)
#int converts the result into a whole number ie. integer
#T/dt = 20/0.5 = 40.0 when we add int(40.0) it gives us 40 
#we take 40 Steps in total 

print(f'Number of steps: {steps}') #which = 40

#now we define our projection function
def project_onto_plane(x):
    return np.dot(e1r, x), np.dot(e2r, x)  # project onto the first two components (e1, e2)  
    """ Returns (e1*x, e2*x) - coordinates of x in the e1-e2 plane """

#we run simulations for all input vectors h_i
fig, ax = plt.subplots(figsize=(8,8)) #all trajectories in one plots to compare
colors_c = ['steelblue', 'salmon', 'lightgreen', 'purple', 'orange']
#one color for every input vector like in Task A and Task B

for i in range(num_inputs):
    h_i = h_list[i] #to pick the i-th input vector while iterating 

    v = np.zeros(N) #starts at 0 with all neurons silent, it is a vector of length 20, resets at 0 with each simulation

    trajectory = [] #empty list to store the (e1, e2) coordinates of our activity vectors at each time step, should contain 40 steps at the end 

    trajectory.append(project_onto_plane(v)) #record initial position and add to list

    for step in range(steps): #this is a nested loop wherein the loop runs 40 times (one per time step starting from 0 to 39)
        v_eq = M @ v + h_i #calculate the equilibrium point for current activity
        #M @ v is matrix multiplication of M and v, gives us the input from the network based on current activity, linearly transformed to give output vectors 
        #+h_i adds the external input 
        #we see with this formula where our networks wants to go in this moment of time 

        v = v_eq + (v - v_eq) * np.exp(-dt)
        #exponential relaxation towards steady state
        #(v-v_eq) is the gap between current state and equilibrium
        #np.exp(-dt) is the exponential decay factor e^(-0.5) = 0.607
        #multiply gap with decay factor and we see the rate of decay each time is 39.3%, so gap reduced this much with every time step
        #we add to v_eq which is the new v that is 40% closer to our steady state 

        trajectory.append(project_onto_plane(v)) #record position at this time step

    #we get the x and y coordinates from trajectory 
    traj_x = [point[0] for point in trajectory] #x coordinate is the first element of each tuple in trajectory
    traj_y = [point[1] for point in trajectory] #y coordinate is the second element of each tuple in trajectory
    #trajectory is a tuple and traj_x pulls out x coordinates and traj_y does this for y 

    #PLOT THE TRAJECTORY 
    col = colors_c[i]
    ax.plot(traj_x, traj_y, color=col, label=f'h{i+1} (neurons {i*2+1}-{i*2+2})')
    ax.scatter(traj_x[0], traj_y[0], color=col, marker='o', s=80) #starting point
    ax.scatter(traj_x[-1], traj_y[-1], color=col, marker='*', s=120) 
    #trajectory_x[-1] is the last point where our network's response will end up 
    #MINUS ONE (-1) ALWAYS MEANS THE LAST ELEMENT IN PYTHON 
    #end point marked with a star and we expect it to land close to v_ss_exact 

    #Reference lines and labels 
    ax.axhline(0, color='lightgray', linewidth=0.8)
    ax.axvline(0, color='lightgray', linewidth=0.8)
    ax.set_xlabel('Projection onto e1')
    ax.set_ylabel('Projection onto e2')
    ax.set_title('Task C - time evolution in e1-e2 plane')
    
    #legends for marker elements and colors 
    from matplotlib.lines import Line2D
    legend_markers = [
        Line2D([0], [0], color='gray', linewidth=1.5,
           label='trajectory path'),
        Line2D([0], [0], marker='o', color='gray',
           label='start (v=0)', markersize=8, linestyle='None'),
        Line2D([0], [0], marker='*', color='gray',
           label='end (≈ v_ss)', markersize=12, linestyle='None'),
    ]
    legend1 = ax.legend(handles=legend_markers, loc='lower right')

    legend2 = ax.legend(loc='upper left')
    # this picks up the colour labels from label= in ax.plot above

    ax.add_artist(legend1)
    # keep both legends visible
    ax.add_artist(legend2)
plt.tight_layout()
plt.show()






#OPTIONAL : TIME EVOLUTION

fig, axes = plt.subplots(num_inputs, 1, figsize=(12, 3 * num_inputs))

for i in range(num_inputs):
    ax  = axes[i]
    h_i = h_list[i]

    v        = np.zeros(N)
    activity = [v.copy()]

    for s in range(steps):
        v_eq = M @ v + h_i
        v    = v_eq + (v - v_eq) * np.exp(-dt)
        activity.append(v.copy())

    activity_array = np.array(activity)
    # shape (41, 20) — 41 time steps, 20 neurons

    # ── Plot each time step as one curve ──────────────────────────────────────
    cmap   = plt.cm.Blues
    # plt.cm.Blues is the colourmap object itself
    # you can call it like a function: cmap(0.3) returns an RGBA colour

    n_curves = len(activity_array)
    # = 41 — one curve per time step

    for t_idx, v_t in enumerate(activity_array):
        colour = cmap(0.15 + 0.85 * t_idx / (n_curves - 1))
        # t_idx goes 0 to 40
        # t_idx / (n_curves-1) goes 0.0 to 1.0
        # 0.15 + 0.85 * that = goes from 0.15 to 1.0
        # we start at 0.15 not 0.0 because cmap(0.0) is nearly white
        # and would be invisible against the white background
        # cmap(value) returns an RGBA colour tuple

        ax.plot(node_index, v_t,
                color=colour,
                linewidth=0.6,
                alpha=0.7)
        # plot this time step's activity across all 20 neurons
        # thin lines (0.6) and slightly transparent (0.7)
        # so many curves can overlap without becoming a solid block

    # ── Draw the input h as a red rectangle ───────────────────────────────────
    ax.step(node_index, h_i,
            where='mid',
            color='red',
            linewidth=1.5,
            label=f'input h{i+1}')
    # step plot of h — flat at zero everywhere except the two stimulated neurons
    # this is the red rectangle your friend has

    # ── Draw the final steady state as a thick dark line ──────────────────────
    ax.plot(node_index, activity_array[-1],
            color='darkblue',
            linewidth=2,
            label='final v_ss')
    # activity_array[-1] = last row = activity at t=20 = steady state
    # drawn thick and dark so it stands out from the lighter curves

    ax.axhline(0, color='lightgray', linewidth=0.5)
    ax.set_ylabel('Activity v(t)')
    ax.set_title(f'h{i+1}: neurons {i*2+1} and {i*2+2} — time evolution')
    ax.legend(loc='upper right', fontsize=8)

axes[-1].set_xlabel('Neuron index k')
plt.suptitle('Task C — time evolution (light=early, dark=late)', fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()