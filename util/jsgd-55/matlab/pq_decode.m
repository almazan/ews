function v = pq_decode (pq, cbase)
    
n = size (cbase, 2);
d = pq.nsq * pq.ds;

v = zeros(d, n, 'single');

for q = 1:pq.nsq
  cents = pq.centroids{q};
  v ((q-1)*pq.ds+1:q*pq.ds, :) = cents(:, 1 + int32(cbase(q, :)));
end 
