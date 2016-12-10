import numpy as np
import time
import operator
 
ratings = np.loadtxt('movieratings.csv', dtype='int', delimiter=',')
movies = np.genfromtxt('movies.csv', delimiter='|',dtype=None)

k = 300
learn = 0.00001
iterations = 200

R = len(ratings)
M = len(movies)
V = np.matrix(np.random.randn(M, k))

#makes co-occurence matrix, X_ij
def make_cooccurence() :
    #matrix to store all user movie entries
    #print "started cooccurence"
    user_movie = np.matrix(np.zeros((1682,943)))
    
    for row in ratings :
        
         user_id = row[1]
         movie_id = row[0] 
         rating = row[2]
         #rating is either 1 or 0
         user_movie[movie_id - 1, user_id - 1]  = rating
    
    #matrix multiplication
    X = np.dot(user_movie, user_movie.T) 
    np.fill_diagonal(X, 0.0)
    #print "max of X", np.amax(X)
    
    #np.savetxt('cooccurrence.csv.gz', X, delimiter=',')
    return X

def train() :
    global V #to modify V
    for i in xrange(iterations):
        #start_time = time.time()
        
        V = V - learn*gradient()
        
        """
        #calulate cost after update
        dot_matrix = np.multiply(V*V.transpose() - X,V*V.transpose() - X)
        np.fill_diagonal(dot_matrix, 0.0)
        cost = np.sum(dot_matrix)
        print cost
        """
                        
        #take Frobenius norm of the inner matrix
        v_s = V*V.T
        np.fill_diagonal(v_s, 0.0) #account for gamma term
        dot_matrix = v_s - X
        norm = np.linalg.norm(dot_matrix, ord=None)
        cost = norm*norm
        if (i < 10): 
            print "iteration", i
            print cost
        

        """
        cost = 0.0
        print "calculating cost"
        for a in range(M):
            for b in range(a + 1, M):
                square = V[a].dot(V[b]) - X[a, b]
                cost += square * square
                
        cost = 2*cost
        print cost
        """
    
        #print "done with iteration", i  
        #print ("in %s seconds" % (time.time() - start_time))
    
def gradient() :
    
    #print "inside gradient"
    
    #size of dot matrix is 1682 by 1682
    v_s = V*V.T
    np.fill_diagonal(v_s, 0.0) #account for gamma term
    dot_matrix = v_s - X
    gradient = 2*dot_matrix*2*V
    return gradient
    
    """
    for i in range(M): 
        inner_col = np.array(np.delete(inner_term[:,i], i, axis=0)) #delete self row
        v_s = np.array(np.delete(V, i, axis=0))
        gradient[i] = np.sum(v_s * inner_col, axis=0) #element wise vector multiplication
    
    gradient *= 4.0
    """
              
    """ 
    for i in range(M):
        for j in range(M):
            if (i != j):
                #elemnt wise vector multiplication 
                v_temp = V[j].reshape(300,)
                print v_temp.shape
                gradient[i] += inner_term[i, j]*(v_temp)
                                       
        gradient[i] *= 4.0
    return gradient
    """
    
def recommend1(movie_id) : 
    
    #calc dot product between this movie's vector and all others
    this_movie = np.matrix(V[movie_id]).reshape(300,1)
    #dim of dots is 1682 by 1
    dots = np.array(np.dot(V, this_movie))
    
    #print this_movie[0:10]
    
    norms = np.zeros(M) #dim of norms is 1682 by 1
    #calc frobenius norm (sqrt(sum of squares)) of each vector
    for i in range(M):
        norms[i] = np.linalg.norm(V[i])
        
    #element wise vector multiplication by scalar
    #print norms[movie_id - 1]
    norms = norms[movie_id - 1]*norms
    norms = norms.reshape(1682, 1)
    norms = np.array(norms)
    
    #calculate cos similarities between this movie and all others
    sim = dots / norms 
    sim.resize(1682,)
    
    #populate dict with key = sim and value = index
    my_dict = {}
    for i in range(len(sim)):
        #print sim[i]
        my_dict[sim[i]] = i
    
    #sort by similarity (the keys)
    sorted_dict = sorted(my_dict.items(), key=operator.itemgetter(0))
    
    #retrieve top 20
    movie_titles = np.array([])
    for ids in range(len(sorted_dict) - 1, len(sorted_dict) - 1 - 20, -1):
        movie_titles = np.append(movie_titles, movies[sorted_dict[ids][1] - 1][1])
         
    return movie_titles

def recommend2(liked) : 
    
    #get average of liked
    store = np.zeros(300).reshape(300,1)
    for i in range(len(liked)):
        store += V[liked[i]].reshape(300,1)
        
    store = store / len(liked)
    #print store[0:10]
                    
    #calc dot product between average vector and all others
    #dim of dots is 1682 by 1
    dots = np.array(np.dot(V, store))
    
    norms = np.zeros(M) #dim of norms is 1682 by 1
    #calc frobenius norm (sqrt(sum of squares)) of each vector
    for i in range(M):
        norms[i] = np.linalg.norm(V[i])
        
    #element wise vector multiplication by scalar
    #print np.linalg.norm(store)
    norms = np.linalg.norm(store)*norms
    norms = norms.reshape(1682, 1)
    norms = np.array(norms)
    
    #calculate cos similarities between this movie and all others
    sim = dots / norms
    sim.resize(1682,)
    
    #populate dict with key = sim and value = index
    my_dict = {}
    for i in range(len(sim)):
        my_dict[sim[i]] = i
    
    #sort by similarity (the keys)
    sorted_dict = sorted(my_dict.items(), key=operator.itemgetter(0))
    
    #retrieve top 20
    movie_titles = np.array([])
    for ids in range(len(sorted_dict) - 1, len(sorted_dict) - 1 - 20, -1):
        movie_titles = np.append(movie_titles, movies[sorted_dict[ids][1] - 1][1])
         
    return movie_titles

X = make_cooccurence()
#print "done with make_cooccurence()"

train()
#print "done with train"

#recommend for Lion King
print recommend1(71)

#recommend for Sleepless in Seattle, Philadelphia Story, and 
#Sex, Lies, and VideoTape
liked = np.array([], dtype=int)
liked = np.append(liked, 88)
liked = np.append(liked, 478)
liked = np.append(liked, 708)
print recommend2(liked)



